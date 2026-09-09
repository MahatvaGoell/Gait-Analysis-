# Template execution contract

## Reference

- Retained DOCX: `C:\Users\MAHATVA GOEL\.codex\plugins\cache\openai-curated-remote\openai-templates\0.1.1\skills\artifact-template-experiment-analysis\assets\reference.docx`
- SHA-256: `D823CD0115186B34C01C6E4B4DA3BE28B64EE73CAC849DBD62D6F4BB6385B0FB`
- Size: 204,643 bytes
- Page count: 7 when exported by Microsoft Word 2024
- Section count: 1
- Reference render: `C:\Users\MAHATVA GOEL\Desktop\Gait Analysis\.codex_tmp_data_explanation\template-reference-render`
- Evidence files: `template-style-evidence.json` and `template_structure.json` in the same task directory

## Page system

- One portrait Letter section, 8.5 x 11 inches.
- Margins are 1.0 inch on all sides.
- Header distance is 0.5 inch; footer distance is 0.5 inch.
- No different first-page or odd/even header treatment.
- Page 1 is a cover. Body paragraph 25 contains the explicit page break.
- Pages 2 onward use the same section, margins, footer, and single-column flow.
- Header part is empty. Footer shows the report name at left and a PAGE field at right.

## Typography

- The reference uses Georgia for all visible content.
- Title style is centered, 40 pt, bold, with 4 pt after.
- Heading 1 is Georgia 18 pt bold, 18 pt before, 4 pt after, and kept with following content.
- Heading 2 is Georgia 16 pt and centered.
- Heading 3 is centered and used for the cover metadata.
- Body text uses the reference normal style with direct Georgia formatting. The final report will use 10.5 to 11 pt body text with natural paragraph spacing.
- The reference uses green headings and title text. The final report will keep the Georgia hierarchy and green decorative cover line/footer, but title and body headings will be black to satisfy the document-authoring title and heading requirement.

## Lists and tables

- Reference lists are stored in `word/numbering.xml`; that part is preserve-only.
- The reference contains eight border-grid tables. Their two-column grids use approximately 2.405/6.945 inches, 2.547/6.803 inches, or 2.263/7.087 inches. Three- and five-column examples divide the 9,350-twip text width deliberately rather than equally when label columns need less space.
- Final tables may be rebuilt inside the editable body while keeping the reference page width and typography. Borders must be light gray `D9D9D9`; header rows will use dark gray fill with white text; alternating body rows may use pale gray.
- Table rows expand to fit wrapped text. Header rows repeat across page breaks. Narrative remains outside tables.

## Components and content flow

- Cover pattern: centered small report label, short green line above the large title, large two-line title, centered report descriptor and date near the lower page, then footer.
- Content pattern: Heading 1 sections, short explanatory paragraphs, compact comparison or metric tables, and numbered/bulleted preparation steps.
- Planned content order: main conclusion, Study 01 objective and classes, source and file organization, measured inventory, file schema, protocol and label construction, quality findings, preparation workflow, evaluation plan, limitations, subject inventory, and protocol code appendix.
- Footer content-control tag `goog_rdk_0` contains the editable report name. The PAGE field remains unchanged.

## Slot map

- `word/document.xml` body paragraph 0, style Heading 2: cover label; rewrite.
- Body paragraphs 1-5: cover spacing and decorative anchored line; preserve structure.
- Body paragraphs 6-7, style Title: cover title lines; rewrite without altering title layout.
- Body paragraphs 8-22: cover spacing; preserve.
- Body paragraphs 23-24, style Heading 3: report descriptor and date; rewrite.
- Body paragraph 25: page-break container; preserve.
- Body paragraphs 26-105 and body tables 0-7: placeholder content; replace with the dataset report. Their semantic patterns may be cloned, but placeholder text must not remain.
- `word/footer1.xml`, content control tag `goog_rdk_0`: replace `Report Name` with `Data Explanation`; preserve the control and its run properties.
- `word/footer1.xml` PAGE field: preserve.
- `word/header1.xml`: preserve unchanged.

## Text coverage and stable locators

- Body text, all eight body tables, empty header paragraphs, footer text, footer PAGE field, the anchored cover line, and the footer content control were inspected.
- There are no footnotes or endnotes. The only field is PAGE in `word/footer1.xml`.
- There are two anchored drawing records in `word/document.xml`, one backed by `word/media/image1.png`; both belong to the cover line and are preserve-only.
- Stable cover locators use paragraph order plus style within `word/document.xml`; footer text uses content-control tag `goog_rdk_0`.
- The replaceable report body starts immediately after the first explicit page break and ends before the section properties element.

## Package preservation inventory

| Part | Bytes | SHA-256 | Treatment |
|---|---:|---|---|
| `[Content_Types].xml` | 1,541 | acab6355fcbf2804d10c17279adc776568ba1f4090e8722c8206380bd0b19ef8 | Preserve |
| `_rels/.rels` | 298 | 77494e8e16bbf29213e494792349e3d49d2397a4007b2816a058337d804e79a1 | Preserve |
| `word/_rels/document.xml.rels` | 1,200 | 3e605c0042be606f29c86b3ce8095c7e28ff70813ad8808e0b6f01ea631bd34c | Preserve |
| `word/_rels/fontTable.xml.rels` | 449 | 18b9c91ee35007ae3d239898d32c7287b7eb5574de0f45216e4cb95ee560d2a6 | Preserve |
| `word/document.xml` | 160,160 | 14193ccf0b7d47c89c2380e058663c0566f172a5ab1299bdefff0a5e140e811f | Edit cover text and body |
| `word/fontTable.xml` | 2,010 | f0ecf22349646bb37f6c6e3a3ad0ea4abcfbef6143653cf98e3fb405ad8656ea | Preserve |
| `word/fonts/NotoSansSymbols-bold.ttf` | 183,344 | afc2a6b7050df884f5deb2618ebbdecb9fa2f163d3ca66ca75f8b069febc6741 | Preserve |
| `word/fonts/NotoSansSymbols-regular.ttf` | 183,620 | 3c9c40311ae6012f7d3c4a4ff98952b2c7bbc0d4c89a93ba22986ff1f1884af5 | Preserve |
| `word/footer1.xml` | 5,532 | 43600114841d1c450b57986bfa5aca3ccfa7a5e9f2135a0e27a71750c44dc986 | Edit report-name control only |
| `word/header1.xml` | 1,791 | a06e51104c1d1da7889e9d733c189bf2267fff63e556f196bbfd45e4a51079b3 | Preserve |
| `word/media/image1.png` | 70 | 18d840af2c50eff9a5241d4b50833a596e6b71af0cee87cf2b3435345f2f7aba | Preserve |
| `word/numbering.xml` | 14,759 | 7e32d1fff31b749d1ca67e8033e2f3b294b6336df93dd0061e50ba466fbd5edb | Preserve |
| `word/settings.xml` | 2,152 | fb088d0a3dbf7f5e5a5a399b898acf16aa2a6fc3b2984bc3eb0d4946f73ac005 | Preserve |
| `word/styles.xml` | 8,499 | e8ce096575d68df59dc88798985b642f7d127ae82bb21f1ea1880f42ac867660 | Preserve |
| `word/theme/theme1.xml` | 7,642 | 742bf86bb312d06389e21644f7e861273fe4f124f422c57cce0c28360b06f32d | Preserve |

## Fidelity gates

- The retained DOCX must retain its recorded SHA-256.
- All package parts except `word/document.xml` and the report-name text in `word/footer1.xml` must match the reference hashes.
- Section count, Letter geometry, margins, cover line, fonts, footer layout, PAGE field, content control, numbering definitions, embedded fonts, and relationships must remain present.
- The final document must render cleanly in Microsoft Word, because the bundled LibreOffice executable is unavailable in the resolved runtime.
- Every final page must be inspected as a PNG for clipping, overflow, broken tables, poor page breaks, and footer placement.
