"""Simple summary exporters."""

from __future__ import annotations

import csv
import json
from io import BytesIO
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as xml_escape
from zipfile import ZIP_DEFLATED, ZipFile

from guard_eval_harness.reports import load_or_build_summary


def _column_name(index: int) -> str:
    """Convert a zero-based column index into spreadsheet letters."""
    letters = []
    current = index + 1
    while current:
        current, remainder = divmod(current - 1, 26)
        letters.append(chr(ord("A") + remainder))
    return "".join(reversed(letters))


def _worksheet_cell(row: int, column: int, value: Any) -> str:
    """Encode a single worksheet cell as XML."""
    reference = f"{_column_name(column)}{row}"
    if value is None:
        return f'<c r="{reference}"/>'
    if isinstance(value, bool):
        numeric = "1" if value else "0"
        return f'<c r="{reference}" t="b"><v>{numeric}</v></c>'
    if isinstance(value, (int, float)):
        return f'<c r="{reference}"><v>{value}</v></c>'
    return (
        f'<c r="{reference}" t="inlineStr"><is><t>'
        f"{xml_escape(str(value))}</t></is></c>"
    )


def _summary_rows(summary: dict[str, Any]) -> list[list[Any]]:
    """Build spreadsheet rows from a rebuilt summary."""
    rows: list[list[Any]] = [
        ["run_name", summary["run_name"]],
        ["status", summary["status"]],
        ["threshold", summary["threshold"]],
        ["adapter", summary.get("adapter")],
        ["model_name", summary.get("model_name")],
        [],
        [
            "display_name",
            "name",
            "sample_count",
            "unsafe_count",
            "count",
            "accuracy",
            "auroc",
            "auprc",
            "precision",
            "recall",
            "f1",
            "fpr",
            "fnr",
            "tp",
            "tn",
            "fp",
            "fn",
        ],
    ]
    for dataset in summary["datasets"]:
        metrics = dataset["metrics"]
        rows.append(
            [
                dataset.get("display_name"),
                dataset["name"],
                dataset.get("sample_count"),
                dataset.get("unsafe_count"),
                metrics.get("count"),
                metrics.get("accuracy"),
                metrics.get("auroc"),
                metrics.get("auprc"),
                metrics.get("precision"),
                metrics.get("recall"),
                metrics.get("f1"),
                metrics.get("fpr"),
                metrics.get("fnr"),
                metrics.get("tp"),
                metrics.get("tn"),
                metrics.get("fp"),
                metrics.get("fn"),
            ]
        )
    return rows


def _worksheet_xml(rows: list[list[Any]]) -> str:
    """Render workbook rows into a single-sheet worksheet XML."""
    xml_rows = []
    for row_index, values in enumerate(rows, start=1):
        cells = [
            _worksheet_cell(row_index, column, value)
            for column, value in enumerate(values)
        ]
        xml_rows.append(
            f'<row r="{row_index}">' + "".join(cells) + "</row>"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/'
        'spreadsheetml/2006/main">'
        "<sheetData>"
        + "".join(xml_rows)
        + "</sheetData></worksheet>"
    )


def _build_xlsx_bytes(summary: dict[str, Any]) -> bytes:
    """Build a minimal XLSX workbook for summary export."""
    rows = _summary_rows(summary)
    worksheet_xml = _worksheet_xml(rows)
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>""",
        )
        archive.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>""",
        )
        archive.writestr(
            "xl/workbook.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="summary" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>""",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>""",
        )
        archive.writestr(
            "xl/styles.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>
  <fills count="1"><fill><patternFill patternType="none"/></fill></fills>
  <borders count="1"><border/></borders>
  <cellStyleXfs count="1"><xf/></cellStyleXfs>
  <cellXfs count="1"><xf xfId="0"/></cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>""",
        )
        archive.writestr("xl/worksheets/sheet1.xml", worksheet_xml)
    return buffer.getvalue()


def export_summary(
    run_dir: str,
    *,
    fmt: str,
    output_path: str,
) -> str:
    """Export a run summary as JSON or CSV."""
    summary = load_or_build_summary(run_dir)
    destination = Path(output_path)

    if fmt == "json":
        destination.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    elif fmt == "csv":
        fieldnames = [
            "name",
            "count",
            "accuracy",
            "auroc",
            "auprc",
            "precision",
            "recall",
            "f1",
            "fpr",
            "fnr",
            "tp",
            "tn",
            "fp",
            "fn",
        ]
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for dataset in summary["datasets"]:
                metrics = dataset["metrics"]
                writer.writerow(
                    {
                        "name": dataset["name"],
                        **{
                            field: metrics.get(field)
                            for field in fieldnames
                            if field != "name"
                        },
                    }
                )
    elif fmt == "xlsx":
        destination.write_bytes(_build_xlsx_bytes(summary))
    else:
        raise ValueError(f"unsupported export format: {fmt}")

    return destination.as_posix()
