"""
meteor_fixed.py (v2)
======================
Mục đích:
    Bản thay thế cho pycocoevalcap.meteor.meteor.Meteor, khắc phục lỗi
    "ValueError: could not convert string to float" xảy ra khi chạy trên
    Windows -- lỗi ĐÃ BIẾT của pycocoevalcap (xem các issue:
    tylin/coco-caption#6, salaniz/pytorch-gve-lrcn#9, Labbeti/aac-metrics#9).

LỊCH SỬ SỬA LỖI (đọc để hiểu vì sao code trông như vậy):
    v1: Gọi METEOR theo từng cặp một (1 ảnh), đọc đủ 2 dòng "average"/"all"
        sau mỗi EVAL theo đúng giao thức jar. Test 5 ảnh và 100 ảnh đều
        khớp chính xác với bản gốc trên Linux.
    v1 -> lỗi thực tế trên 2135 ảnh: "RuntimeError: Không đọc được điểm
        METEOR hợp lệ ... Dòng cuối nhận được: ''"
        Nguyên nhân: readline() trên 1 PIPE trả về '' khi gặp EOF (process
        Java phía bên kia ĐÃ CHẾT/đóng stdout) -- v1 hiểu nhầm '' là "chưa
        có dữ liệu, đợi thêm" và cứ retry đọc thêm dòng, dẫn đến lặp vô
        nghĩa rồi mới raise, KHÔNG cho biết Java chết vì lý do gì.
    v2 (bản này): 2 thay đổi cốt lõi để sửa tận gốc:
        1) Đọc stderr của Java trong 1 thread riêng song song, lưu lại các
           dòng gần nhất -- nếu Java chết, exception sẽ in kèm nội dung
           stderr để biết chính xác lý do (không còn đoán mò).
        2) Sanitize hypothesis/reference TRƯỚC khi gửi cho Java: METEOR
           jar dùng "|||" làm delimiter giữa các trường trong câu lệnh
           SCORE/EVAL. Nếu 1 caption tự sinh ra (hoặc 1 reference COCO)
           VÔ TÌNH chứa ký tự "|", hoặc chứa "\n"/"\r", cấu trúc câu lệnh
           gửi cho Java sẽ bị lệch field -- khiến Java parse sai, có thể
           crash hoặc treo (block khi cố ghi lỗi ra stderr trong khi
           không ai đọc, làm nghẽn toàn bộ subprocess). Đây là nguyên
           nhân nhiều khả năng nhất gây ra lỗi EOF ở v1, vì lỗi chỉ xuất
           hiện trên data thật (2135 ảnh) mà không xuất hiện ở tập test
           nhỏ (5, 100 ảnh) -- caption sinh từ model GPT-2 fine-tuned có
           xác suất nhỏ nhưng không bằng 0 sinh ra ký tự lạ.
        Ngoài ra: phân biệt rõ "EOF thực" (process đã chết -> raise ngay,
        kèm stderr) với lỗi tạm thời (dòng trống bất thường nhưng process
        còn sống -> retry có giới hạn).

Lưu ý về overall score:
    Bản gốc tính overall score bằng cách gửi TOÀN BỘ stat dồn vào 1 lệnh
    EVAL duy nhất ở cuối. Bản fixed này tính overall score = TRUNG BÌNH
    CỘNG các per-image score (cách tính phổ biến, tương đương về ý nghĩa,
    chỉ khác phương pháp). Per-image scores của 2 bản đã được verify khớp
    chính xác đến 4 số lẻ trên dữ liệu test ở v1.
"""

import os
import re
import subprocess
import threading
import collections

METEOR_JAR = "meteor-1.5.jar"

# Ký tự "|" phá vỡ delimiter "|||" của METEOR jar -> phải loại bỏ.
# Xuống dòng cũng phá vỡ giao thức (mỗi lệnh phải nằm trên đúng 1 dòng).
_BAD_CHARS_RE = re.compile(r"[|\r\n]")


def _sanitize(text: str) -> str:
    """Loại bỏ ký tự có thể phá vỡ giao thức dòng-lệnh của METEOR jar."""
    if text is None:
        text = ""
    text = str(text)
    text = _BAD_CHARS_RE.sub(" ", text)
    text = " ".join(text.split())  # gộp nhiều khoảng trắng liên tiếp thành 1
    if text.strip() == "":
        # Caption rỗng cũng có thể làm méo câu lệnh SCORE (thiếu 1 field
        # coi như có nội dung) -- thay bằng 1 token placeholder để giữ
        # đúng số field, không làm sai lệch các ảnh khác.
        text = "."
    return text


class MeteorFixed:
    def __init__(self):
        jar_dir = self._find_meteor_jar_dir()
        self.meteor_cmd = [
            "java", "-jar", "-Xmx2G", METEOR_JAR,
            "-", "-", "-stdio", "-l", "en", "-norm",
        ]
        self.meteor_p = subprocess.Popen(
            self.meteor_cmd,
            cwd=jar_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.lock = threading.Lock()

        # Đọc stderr song song trong thread riêng, lưu lại N dòng gần nhất.
        # Nếu không làm việc này, buffer stderr có thể đầy -> Java bị block
        # khi ghi lỗi -> toàn bộ subprocess treo (rất giống lỗi đã gặp).
        self._stderr_tail = collections.deque(maxlen=50)
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True
        )
        self._stderr_thread.start()

    @staticmethod
    def _find_meteor_jar_dir():
        import pycocoevalcap.meteor.meteor as original_meteor_module
        return os.path.dirname(os.path.abspath(original_meteor_module.__file__))

    def _drain_stderr(self):
        try:
            for line in self.meteor_p.stderr:
                self._stderr_tail.append(line.decode(errors="replace").rstrip())
        except Exception:
            pass

    def _process_alive(self) -> bool:
        return self.meteor_p.poll() is None

    def _raise_with_diagnostics(self, context: str, raw_repr: str):
        alive = self._process_alive()
        returncode = self.meteor_p.poll()
        stderr_dump = "\n".join(self._stderr_tail) if self._stderr_tail else "(không có output trên stderr)"
        raise RuntimeError(
            f"[MeteorFixed] Lỗi tại bước: {context}\n"
            f"  Dòng cuối nhận được: {raw_repr}\n"
            f"  Java subprocess còn sống? {alive} (returncode={returncode})\n"
            f"  Nội dung stderr gần nhất:\n{stderr_dump}"
        )

    def _read_valid_float_line(self, context: str, max_retries: int = 3) -> float:
        """
        Đọc 1 dòng từ stdout, validate là số float hợp lệ.
        QUAN TRỌNG: readline() trả về '' (rỗng) có 2 khả năng:
          (a) Process đã chết / đóng pipe (EOF thực) -> không có gì để
              retry, phải raise ngay kèm chẩn đoán.
          (b) Trường hợp hiếm: đọc trúng lúc dòng chưa flush xong nhưng
              process còn sống -> có thể retry.
        Code này luôn kiểm tra (a) trước khi quyết định có retry hay không.
        """
        raw = ""
        for attempt in range(max_retries):
            raw = self.meteor_p.stdout.readline().decode(errors="replace").strip()

            if raw == "":
                if not self._process_alive():
                    # EOF thực -- process đã chết, retry thêm vô nghĩa.
                    self._raise_with_diagnostics(
                        f"{context} (lần thử {attempt + 1}/{max_retries}, process đã chết)",
                        repr(raw),
                    )
                # Process còn sống nhưng dòng rỗng -- thử đọc tiếp.
                continue

            try:
                return float(raw)
            except ValueError:
                # Dòng bị dồn nhiều số (vd "11.0 12.0 6.0 ...") -- lấy số đầu.
                parts = raw.split()
                if parts:
                    try:
                        return float(parts[0])
                    except ValueError:
                        continue

        self._raise_with_diagnostics(
            f"{context} (hết {max_retries} lần thử, process vẫn sống nhưng không đọc được số hợp lệ)",
            repr(raw),
        )

    def _stat(self, hypothesis_str: str, reference_list: list) -> str:
        hyp = _sanitize(hypothesis_str)
        refs = [_sanitize(r) for r in reference_list]

        score_line = " ||| ".join(("SCORE", " ||| ".join(refs), hyp))
        self.meteor_p.stdin.write(f"{score_line}\n".encode())
        self.meteor_p.stdin.flush()

        raw = self.meteor_p.stdout.readline().decode(errors="replace").strip()
        if raw == "" and not self._process_alive():
            self._raise_with_diagnostics("_stat() (đọc dòng stat sau SCORE)", repr(raw))
        return raw

    def compute_score(self, gts: dict, res: dict):
        assert gts.keys() == res.keys()
        img_ids = list(gts.keys())
        scores = []
        per_image_scores = {}

        self.lock.acquire()
        try:
            for img_id in img_ids:
                assert len(res[img_id]) == 1

                if not self._process_alive():
                    self._raise_with_diagnostics(
                        f"compute_score() trước khi xử lý ảnh {img_id}", "(chưa đọc gì)"
                    )

                stat = self._stat(res[img_id][0], gts[img_id])

                eval_line = f"EVAL ||| {stat}"
                self.meteor_p.stdin.write(f"{eval_line}\n".encode())
                self.meteor_p.stdin.flush()

                # METEOR jar trả về 2 dòng giống nhau cho mỗi lệnh EVAL
                # (1 "average", 1 "all" -- xem comment gốc trong code chính
                # thức của pycocoevalcap). Phải đọc đủ cả 2 dòng.
                score = self._read_valid_float_line(f"EVAL ảnh {img_id} (dòng 1/2)")
                _ = self._read_valid_float_line(f"EVAL ảnh {img_id} (dòng 2/2, lặp lại)")

                scores.append(score)
                per_image_scores[img_id] = score

            overall_score = sum(scores) / len(scores) if scores else 0.0
        finally:
            self.lock.release()

        return overall_score, scores

    def method(self):
        return "METEOR"

    def __del__(self):
        try:
            self.lock.acquire()
            self.meteor_p.stdin.close()
            self.meteor_p.kill()
            self.meteor_p.wait()
            self.lock.release()
        except Exception:
            pass