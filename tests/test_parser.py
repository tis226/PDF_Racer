import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PDF_Parser_V2_patched_full import Line, _parse_qas_from_lines


def make_line(text: str, order: int, column: str = "left", page: int = 0) -> Line:
    top = 800 - order * 20
    return Line(
        text=text,
        x0=0.0,
        y0=top - 10.0,
        x1=100.0,
        y1=top,
        size=10.0,
        font="TestFont",
        column=column,
        page_index=page,
    )


def extract_question(qas, number):
    for qa in qas:
        if qa["content"]["question_number"] == number:
            return qa
    raise AssertionError(f"Question {number} not found")


def test_inline_options_removed_from_question_text():
    lines = [
        make_line(
            "15. Inline question text ① 첫 번째 보기 ② 두 번째 보기 ③ 세 번째 보기 ④ 네 번째 보기",
            0,
        ),
        make_line("16. Separate question", 1),
        make_line("① alpha", 2),
        make_line("② beta", 3),
        make_line("③ gamma", 4),
        make_line("④ delta", 5),
    ]

    qas = _parse_qas_from_lines(lines, subject="형법", year=2022, target="default")

    q15 = extract_question(qas, 15)
    content15 = q15["content"]
    assert content15["question_text"] == "Inline question text"
    option_labels = [opt["index"] for opt in content15["options"]]
    assert option_labels == ["①", "②", "③", "④"]
    assert all("보기" in opt["text"] for opt in content15["options"])

    q16 = extract_question(qas, 16)
    assert q16["content"]["question_text"] == "Separate question"


def test_option_trailing_question_number_split_into_new_question():
    lines = [
        make_line("19. Prev question stem", 0),
        make_line("① alpha", 1),
        make_line("② beta", 2),
        make_line("③ gamma", 3),
        make_line("④ delta 끝부분 20. 다음 문제 서두", 4),
        make_line("㉠ 진술 하나", 5),
        make_line("㉡ 진술 둘", 6),
        make_line("① 1개", 7),
        make_line("② 2개", 8),
        make_line("③ 3개", 9),
        make_line("④ 4개", 10),
    ]

    qas = _parse_qas_from_lines(lines, subject="형법", year=2022, target="default")

    q19 = extract_question(qas, 19)
    opts19 = {opt["index"]: opt["text"] for opt in q19["content"]["options"]}
    assert "20." not in opts19["④"]
    assert "끝부분" in opts19["④"]

    q20 = extract_question(qas, 20)
    text20 = q20["content"]["question_text"]
    assert text20.startswith("다음 문제 서두")
    assert "끝부분" not in text20
    assert "㉠" in text20 and "㉡" in text20
    labels20 = [opt["index"] for opt in q20["content"]["options"]]
    assert labels20 == ["①", "②", "③", "④"]
