"""只读检查 2026 APMCM C 题提交文件。"""

import argparse
import re
import sys
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile
from zlib import error as ZipStreamError

import fitz


MAX_PAPER_BYTES = 50_000_000
MAX_NON_APPENDIX_PAGES = 25
AI_FILENAME = "AI工具使用详情.pdf"
HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
TEAM_PATTERN = re.compile(r"apmcm263[0-9]{5}", re.IGNORECASE)


def check_pdf(source, label, errors):
    try:
        pdf = (
            fitz.open(stream=source, filetype="pdf")
            if isinstance(source, bytes)
            else fitz.open(source)
        )
    except (RuntimeError, OSError, ValueError) as exc:
        errors.append(f"{label} 无法打开（{exc}）")
        return None
    if not pdf.is_pdf or pdf.needs_pass or not pdf.page_count or pdf.is_repaired:
        errors.append(f"{label} 需要密码、需要修复、没有页面或格式有误")
        pdf.close()
        return None
    return pdf


def check_paper(path, stem, team, appendix_start, errors, review):
    if path.name != f"{stem}.pdf":
        errors.append(f"论文应命名为 {stem}.pdf")
    if path.is_file() and path.stat().st_size > MAX_PAPER_BYTES:
        errors.append("论文 PDF 大于 50 MB（按 50,000,000 字节预检）")
    pdf = check_pdf(path, "论文 PDF", errors)
    if pdf is None:
        return
    with pdf:
        if pdf.page_count < 3:
            errors.append("论文至少应有摘要、目录及正文三页")
        if appendix_start is not None and not 4 <= appendix_start <= pdf.page_count:
            errors.append("附录首个 PDF 页码须在第 4 页与文件末页之间")
            appendix_start = None
        non_appendix = pdf.page_count if appendix_start is None else appendix_start - 1
        if non_appendix > MAX_NON_APPENDIX_PAGES:
            review.append(
                f"非附录暂计 {non_appendix} 页。请确认附录起页及官方主文 25 页的范围"
            )
        if appendix_start and not re.search(
            r"\bappendi(?:x|ces)\b", pdf[appendix_start - 1].get_text(), re.IGNORECASE
        ):
            review.append(f"请核对 PDF 第 {appendix_start} 页确为附录起页")
        first_page = pdf[0].get_text()
        if not re.search(r"\b(summary|abstract)\b", first_page, re.IGNORECASE):
            review.append("第 1 页未检出 Summary 或 Abstract 标签，请人工核对")
        if not re.search(
            r"\b(?:problem(?:\s+(?:chosen|selected))?|problem\s+number)"
            r"\s*[:\s]*C\b",
            first_page,
            re.IGNORECASE,
        ):
            review.append("第 1 页未检出可识别的选题 C 标记，请人工核对")
        if pdf.page_count >= 2 and not re.search(
            r"\b(contents|table\s+of\s+contents)\b", pdf[1].get_text(), re.IGNORECASE
        ):
            review.append("第 2 页未检出 Contents 标签，请人工核对")

        missing_header, no_text, han_pages = [], [], []
        for number, page in enumerate(pdf, 1):
            page_text = page.get_text()
            if number <= non_appendix and HAN.search(page_text):
                han_pages.append(number)
            page_area = page.rect
            header_bottom = page_area.y0 + min(100, page_area.height / 5)
            top = fitz.Rect(page_area.x0, page_area.y0, page_area.x1, header_bottom)
            body = fitz.Rect(
                page_area.x0, header_bottom, page_area.x1, page_area.y1 - 45
            )
            if not page.get_text(clip=body).strip():
                no_text.append(number)
            header = page.get_text(clip=top)
            positions = re.findall(r"\bpage\s+(\d+)\b", header, re.IGNORECASE)
            if team.lower() not in header.lower() or str(number) not in positions:
                missing_header.append(number)
        if missing_header:
            review.append(f"请核对这些 PDF 页顶部的队号或页码 {missing_header}")
        if no_text:
            review.append(f"这些 PDF 页正文区无可抽取文本，请核对图像页或空白页 {no_text}")
        if han_pages:
            review.append(f"这些非附录 PDF 页抽取到汉字，请核对正文与图表语言 {han_pages}")
        print(f"论文 PDF 共 {pdf.page_count} 页，当前口径非附录 {non_appendix} 页")


def check_control(path, stem, errors):
    suffix = path.suffix.lower()
    if suffix not in {".pdf", ".jpg", ".png"} or path.stem != f"{stem}cns":
        errors.append(f"承诺书文件名应为 {stem}cns.pdf、.jpg 或 .png")
    try:
        if suffix == ".pdf":
            pdf = check_pdf(path, "承诺书", errors)
            if pdf is not None:
                with pdf:
                    if pdf.page_count != 1:
                        errors.append("承诺书 PDF 应为单独一页")
        elif suffix in {".jpg", ".png"}:
            with path.open("rb") as handle:
                signature = handle.read(8)
            expected = b"\x89PNG\r\n\x1a\n" if suffix == ".png" else b"\xff\xd8\xff"
            if not signature.startswith(expected):
                errors.append("承诺书图片内容与扩展名不符")
            else:
                fitz.Pixmap(str(path))
    except (OSError, RuntimeError, ValueError) as exc:
        errors.append(f"承诺书图片无法打开（{exc}）")


def check_support(path, stem, originals, errors, review):
    if path.stem != f"{stem}fj" or path.suffix.lower() not in {".zip", ".rar"}:
        errors.append(f"支撑材料应命名为 {stem}fj.zip 或 {stem}fj.rar")
    if path.suffix.lower() == ".rar":
        if not path.is_file():
            errors.append("支撑材料 RAR 文件不存在")
        else:
            review.append("请在本地解压验证 RAR 和包内文件，脚本内置 ZIP 检查")
        return False
    try:
        with ZipFile(path) as archive:
            names = [item.filename for item in archive.infolist() if not item.is_dir()]
            if not names:
                errors.append("支撑材料 ZIP 为空")
                return False
            corrupt = archive.testzip()
            if corrupt:
                errors.append(f"支撑材料 ZIP 校验失败（{corrupt}）")
                return False
            basenames = [PurePosixPath(name.replace("\\", "/")).name for name in names]
            matching = sorted({name for name in basenames if name.casefold() in originals})
            if matching:
                errors.append("支撑包发现与比赛原始附件重名的文件 " + "、".join(matching))
            ai_paths = [name for name, base in zip(names, basenames) if base == AI_FILENAME]
            if len(ai_paths) > 1:
                errors.append("支撑包中有多份同名 AI 使用说明")
            if ai_paths:
                pdf = check_pdf(archive.read(ai_paths[0]), "包内 AI 使用说明", errors)
                if pdf is not None:
                    pdf.close()
                    return True
    except (BadZipFile, RuntimeError, OSError, ValueError, EOFError, ZipStreamError) as exc:
        errors.append(f"支撑材料 ZIP 无法通过解压校验（{exc}）")
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team", required=True, help="赛氪完整队号")
    parser.add_argument("--paper", type=Path, required=True, help="英文论文 PDF")
    parser.add_argument("--control", type=Path, required=True, help="已签名的独立承诺书")
    parser.add_argument("--support", type=Path, help="已有支撑包 ZIP 或 RAR")
    parser.add_argument("--ai-details", type=Path, help="单独提供的 AI 使用说明 PDF")
    parser.add_argument("--used-ai", action="store_true", help="参赛作品采用 AI 辅助")
    parser.add_argument("--appendix-start", type=int, help="PDF 中附录起始页，从 1 开始")
    args = parser.parse_args()
    if not TEAM_PATTERN.fullmatch(args.team):
        parser.error("请填写 apmcm263 加 5 位数字的完整赛氪队号")

    team = args.team
    stem = "C" + team
    errors, review = [], []
    check_paper(args.paper, stem, team, args.appendix_start, errors, review)
    check_control(args.control, stem, errors)

    ai_present = False
    if args.support:
        project = Path(__file__).resolve().parent
        originals = {path.name.casefold() for path in (project / "date").iterdir() if path.is_file()}
        originals.add("supplementary4_signaldegradationestimation.pdf")
        ai_present = check_support(args.support, stem, originals, errors, review)
    if args.ai_details:
        if args.ai_details.name != AI_FILENAME:
            errors.append(f"AI 使用说明应命名为 {AI_FILENAME}")
        pdf = check_pdf(args.ai_details, "AI 使用说明", errors)
        if pdf is not None:
            pdf.close()
            ai_present = True
    if args.used_ai and not ai_present:
        errors.append(f"请提供 {AI_FILENAME}，可用 --ai-details 指定待检查的独立文件")

    for item in errors:
        print(f"需修正 · {item}", file=sys.stderr)
    for item in review:
        print(f"需复核 · {item}", file=sys.stderr)
    print("人工核对清单包括签名、匿名、原生稿与导出稿一致性、图表语言、平台栏位和提交回执。")
    if errors:
        return 2
    if review:
        return 1
    print("当前文件通过脚本的机械检查。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
