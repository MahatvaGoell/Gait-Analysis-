"""Fill the IEEE A4 conference Word template with the paper (same content as paper/main.tex).

Usage:  python paper/make_docx.py <template.docx> <figure_dir> <out.docx>
The figure dir must hold flow.png and setup.png; the other figures come from paper/figures.
The template is a strict-OOXML IEEE template; its styles (paper title, Author, Abstract, Heading 1/2/5,
Body Text, bullet list, table head, figure caption, references) and automatic numbering are reused.
"""
import re
import shutil
import sys
import zipfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import content as c  # noqa: E402

TEMPLATE, FIGDIR, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
PAPER_FIGS = Path(__file__).resolve().parent / "figures"
COL_IN = 3.38   # column width in inches (A4, IEEE margins, 18 pt gap)

with zipfile.ZipFile(TEMPLATE) as z:
    files = {n: z.read(n) for n in z.namelist()}
doc = files["word/document.xml"].decode("utf-8")
head = doc[:doc.index("<w:body>") + len("<w:body>")]
body = doc[doc.index("<w:body>") + len("<w:body>"):doc.index("</w:body>")]
sections = re.findall(r"<w:sectPr.*?</w:sectPr>", body, re.S)
title_sect, body_sect, final_sect = sections[0], sections[-2], sections[-1]
title_sect = title_sect.replace('<w:footerReference w:type="first" r:id="rId8"/>', '<w:footerReference w:type="default" r:id="rId8"/><w:footerReference w:type="first" r:id="rId8"/>')
author_sect = sections[1].replace('<w:cols w:num="3" w:space="36pt"/>', '<w:cols w:num="2" w:space="36pt"/>')


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def runs(text, base=""):
    """Tiny markup: *italic*, **bold**, _{sub}, ^{sup}, {{red placeholder}}."""
    out = []
    for tok in re.split(r"(\*\*.+?\*\*|\*.+?\*|_\{.+?\}|\^\{.+?\}|\{\{.+?\}\})", text):
        if not tok:
            continue
        rpr, t = base, tok
        if tok.startswith("**"):
            rpr, t = base + "<w:b/>", tok[2:-2]
        elif tok.startswith("*"):
            rpr, t = base + "<w:i/><w:iCs/>", tok[1:-1]
        elif tok.startswith("_{"):
            rpr, t = base + '<w:vertAlign w:val="subscript"/>', tok[2:-1]
        elif tok.startswith("^{"):
            rpr, t = base + '<w:vertAlign w:val="superscript"/>', tok[2:-1]
        elif tok.startswith("{{"):
            rpr, t = base + '<w:color w:val="FF0000"/>', "[" + tok[2:-2] + "]"
        out.append(f'<w:r>{"<w:rPr>" + rpr + "</w:rPr>" if rpr else ""}<w:t xml:space="preserve">{esc(t)}</w:t></w:r>')
    return "".join(out)


def p(style, text="", ppr="", raw_runs=None):
    return (f'<w:p><w:pPr><w:pStyle w:val="{style}"/>{ppr}</w:pPr>'
            f'{raw_runs if raw_runs is not None else runs(text)}</w:p>')


rels, images = [], 0


def figure(path, width_in, caption):
    global images
    images += 1
    rid, name = f"rIdImg{images}", f"image{images}.png"
    files[f"word/media/{name}"] = Path(path).read_bytes()
    rels.append(f'<Relationship Id="{rid}" Type="http://purl.oclc.org/ooxml/officeDocument/relationships/image" Target="media/{name}"/>')
    w, h = Image.open(path).size
    cx = int(width_in * 914400); cy = int(cx * h / w)
    drawing = (f'<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
               f'<wp:docPr id="{100 + images}" name="Figure {images}"/>'
               f'<a:graphic xmlns:a="http://purl.oclc.org/ooxml/drawingml/main"><a:graphicData uri="http://purl.oclc.org/ooxml/drawingml/picture">'
               f'<pic:pic xmlns:pic="http://purl.oclc.org/ooxml/drawingml/picture"><pic:nvPicPr><pic:cNvPr id="{100 + images}" name="{name}"/><pic:cNvPicPr/></pic:nvPicPr>'
               f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
               f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
               f'</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>')
    return (f'<w:p><w:pPr><w:keepNext/><w:spacing w:before="6pt" w:after="0pt"/><w:jc w:val="center"/></w:pPr>{drawing}</w:p>'
            + p("figurecaption", caption))


TBL_PR = ('<w:tblPr><w:tblW w:w="0pt" w:type="dxa"/><w:jc w:val="center"/><w:tblBorders>'
          '<w:top w:val="single" w:sz="4" w:space="0" w:color="auto"/><w:bottom w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
          '<w:insideH w:val="single" w:sz="2" w:space="0" w:color="auto"/><w:start w:val="single" w:sz="2" w:space="0" w:color="auto"/>'
          '<w:end w:val="single" w:sz="2" w:space="0" w:color="auto"/><w:insideV w:val="single" w:sz="2" w:space="0" w:color="auto"/>'
          '</w:tblBorders><w:tblLayout w:type="fixed"/><w:tblCellMar><w:start w:w="2pt" w:type="dxa"/><w:end w:w="2pt" w:type="dxa"/></w:tblCellMar></w:tblPr>')


def table(caption, widths_pt, rows, header_rows=1, bold_cells=()):
    out = [p("tablehead", caption)]
    grid = "".join(f'<w:gridCol w:w="{int(w * 20)}"/>' for w in widths_pt)
    trs = []
    for r, row in enumerate(rows):
        tcs = []
        for c, cell in enumerate(row):
            span = 1
            if isinstance(cell, tuple):
                cell, span = cell
            width = sum(widths_pt[c:c + span]) if span > 1 else widths_pt[c]
            style = "tablecolhead" if r < header_rows else "tablecopy"
            jc = '<w:jc w:val="center"/>' if (c > 0 or r < header_rows) else '<w:jc w:val="start"/>'
            txt = runs(cell, "<w:b/>" if (r, c) in bold_cells else "")
            span_xml = '<w:gridSpan w:val="%d"/>' % span if span > 1 else ""
            tcs.append(f'<w:tc><w:tcPr><w:tcW w:w="{width}pt" w:type="dxa"/>{span_xml}<w:vAlign w:val="center"/></w:tcPr>'
                       f'<w:p><w:pPr><w:pStyle w:val="{style}"/>{jc}</w:pPr>{txt}</w:p></w:tc>')
        trs.append(f'<w:tr><w:trPr><w:cantSplit/>{"<w:tblHeader/>" if r < header_rows else ""}<w:jc w:val="center"/></w:trPr>{"".join(tcs)}</w:tr>')
    out.append(f'<w:tbl>{TBL_PR}<w:tblGrid>{grid}</w:tblGrid>{"".join(trs)}</w:tbl>')
    out.append('<w:p><w:pPr><w:spacing w:before="0pt" w:after="0pt"/></w:pPr></w:p>')
    return "".join(out)


def author(name, lines, column_break=False):
    rs = runs(name, "<w:sz w:val=\"18\"/><w:szCs w:val=\"18\"/>")
    for k, line in enumerate(lines):
        italic = "<w:i/>" if k < 2 else ""
        rs += f'<w:r><w:rPr><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr><w:br/></w:r>' + runs(line, italic + '<w:sz w:val="18"/><w:szCs w:val="18"/>')
    if column_break:   # the next author starts at the top of the second column
        rs += '<w:r><w:br w:type="column"/></w:r>'
    return p("Author", raw_runs=rs, ppr='<w:spacing w:before="5pt" w:beforeAutospacing="1"/>')


B = "BodyText"
parts = []
parts.append(p("papertitle", c.TITLE,
               '<w:spacing w:before="5pt" w:beforeAutospacing="1" w:after="5pt" w:afterAutospacing="1"/>'))
parts.append(f'<w:p><w:pPr><w:pStyle w:val="Author"/><w:spacing w:before="5pt" w:beforeAutospacing="1" w:after="5pt" w:afterAutospacing="1" w:line="6pt" w:lineRule="auto"/>{title_sect}</w:pPr></w:p>')
# Four authors in one row (borderless table), same line breaks as the LaTeX version.
author_lines = c.AUTHORS
def author_cell(name, lines):
    rs = runs(name, '<w:sz w:val="22"/><w:szCs w:val="22"/>')
    for line in lines:
        rs += '<w:r><w:rPr><w:sz w:val="17"/></w:rPr><w:br/></w:r>' + runs(line, '<w:sz w:val="17"/><w:szCs w:val="17"/>')
    para = f'<w:p><w:pPr><w:pStyle w:val="Author"/><w:spacing w:before="0pt" w:after="0pt"/><w:jc w:val="center"/></w:pPr>{rs}</w:p>'
    return f'<w:tc><w:tcPr><w:tcW w:w="126pt" w:type="dxa"/></w:tcPr>{para}</w:tc>'
parts.append('<w:tbl><w:tblPr><w:tblW w:w="504pt" w:type="dxa"/><w:jc w:val="center"/><w:tblBorders><w:top w:val="nil"/><w:start w:val="nil"/>'
             '<w:bottom w:val="nil"/><w:end w:val="nil"/><w:insideH w:val="nil"/><w:insideV w:val="nil"/></w:tblBorders><w:tblLayout w:type="fixed"/>'
             '<w:tblCellMar><w:start w:w="1pt" w:type="dxa"/><w:end w:w="1pt" w:type="dxa"/></w:tblCellMar></w:tblPr>'
             '<w:tblGrid>' + '<w:gridCol w:w="2520"/>' * 4 + '</w:tblGrid><w:tr><w:trPr><w:cantSplit/><w:jc w:val="center"/></w:trPr>'
             + "".join(author_cell(n, l) for n, l in author_lines) + '</w:tr></w:tbl>')
parts.append('<w:p><w:pPr><w:spacing w:before="0pt" w:after="0pt" w:line="14pt" w:lineRule="exact"/></w:pPr></w:p>')   # gap before the abstract
parts.append(f'<w:p><w:pPr><w:spacing w:before="0pt" w:after="12pt"/>{author_sect.replace(chr(60) + "w:cols w:num=" + chr(34) + "2" + chr(34) + " w:space=" + chr(34) + "36pt" + chr(34) + "/>", "<w:cols w:space=" + chr(34) + "36pt" + chr(34) + "/>")}</w:pPr></w:p>')

KERNEL = "*K*(**x**, **x**\u2032) = exp(\u2212*\u03b3*\u2016**x** \u2212 **x**\u2032\u2016^{2})"
FIG_FILES = {"flow": FIGDIR / "flow.png", "setup": FIGDIR / "setup.png",
             "seg": PAPER_FIGS / "fig1_segmentation.png", "confusion": PAPER_FIGS / "fig_confusion.png"}
TABLES = {"t1": ([56, 48, 90, 50], c.T1, 1, c.T1_BOLD), "t2": ([68, 29, 29, 29, 29, 29, 29], c.T2, 2, set()), "t3": ([130, 50, 50], c.T3, 1, c.T3_BOLD)}

parts.append(p("Abstract", raw_runs='<w:r><w:rPr><w:i/><w:iCs/></w:rPr><w:t>Abstract</w:t></w:r><w:r><w:t>\u2014</w:t></w:r>' + runs(c.ABSTRACT)))
parts.append(p("Keywords", raw_runs='<w:r><w:t>Index Terms\u2014</w:t></w:r>' + runs(c.KEYWORDS)))
for kind, value in c.BLOCKS:
    if kind == "h1":
        parts.append(p("Heading1", value))
    elif kind == "h2":
        parts.append(p("Heading2", value))
    elif kind == "p":
        parts.append(p(B, value.replace("{{KERNEL}}", KERNEL)))
    elif kind == "bullet":
        parts.append(p("bulletlist", value))
    elif kind == "fig":
        parts.append(figure(FIG_FILES[value], COL_IN, c.CAPTIONS[value]))
    elif kind == "table":
        widths, rows, head_rows, bold = TABLES[value]
        parts.append(table(c.TABLE_TITLES[value], widths, rows, header_rows=head_rows, bold_cells=bold))
parts.append(p("Heading5", "References"))
for ref in c.REFERENCES:
    parts.append(p("references", ref, '<w:ind w:start="17.70pt" w:hanging="17.70pt"/>'))
parts.append(f'<w:p><w:pPr>{body_sect}</w:pPr></w:p>')

ftr = files["word/footer1.xml"].decode("utf-8")
ftr = ftr[:ftr.index(">", ftr.index("<w:ftr")) + 1] + (
    '<w:p><w:pPr><w:pStyle w:val="Footer"/><w:jc w:val="center"/></w:pPr><w:r><w:fldChar w:fldCharType="begin"/></w:r>'
    '<w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r><w:r><w:fldChar w:fldCharType="separate"/></w:r>'
    '<w:r><w:t>1</w:t></w:r><w:r><w:fldChar w:fldCharType="end"/></w:r></w:p></w:ftr>')
files["word/footer1.xml"] = ftr.encode("utf-8")
files["word/document.xml"] = (head + "".join(parts) + final_sect + "</w:body></w:document>").encode("utf-8")
rel_xml = files["word/_rels/document.xml.rels"].decode("utf-8")
rel_xml = re.sub(r'<Relationship Id="rId(9|10)" [^>]*/>', "", rel_xml)   # template's example hyperlinks
files["word/_rels/document.xml.rels"] = rel_xml.replace("</Relationships>", "".join(rels) + "</Relationships>").encode("utf-8")
ct = files["[Content_Types].xml"].decode("utf-8")
if 'Extension="png"' not in ct:
    ct = ct.replace("<Default Extension=\"xml\"", '<Default Extension="png" ContentType="image/png"/><Default Extension="xml"')
files["[Content_Types].xml"] = ct.encode("utf-8")
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for name, data in files.items():
        z.writestr(name, data)
print("written", OUT, "images", images)
