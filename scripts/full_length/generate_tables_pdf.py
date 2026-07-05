#!/usr/bin/env python3
"""
PhaseFlow 论文表格 PDF 生成器
使用 fpdf2 生成专业的论文级别 PDF 表格

Usage:
    python scripts/generate_tables_pdf.py --task all
"""

from fpdf import FPDF
from pathlib import Path
import sys

# 颜色定义 (符合论文配色)
HEADER_BG = (86, 102, 158)      # #56669E 深靛蓝
HEADER_TEXT = (255, 255, 255)      # 白色
ALT_ROW_1 = (245, 245, 245)       # 浅灰斑马纹
ALT_ROW_2 = (255, 255, 255)      # 白色
BORDER_COLOR = (77, 82, 88)       # #4D5258
TEXT_COLOR = (51, 51, 51)         # #333333
ACCENT = (201, 133, 123)          # #C9857B 灰珊瑚

# 默认输出目录
DEFAULT_LLPS_DIR = './artifacts/data/llps/figures'
DEFAULT_DPR_DIR = './artifacts/data/dpr/figures'


class PaperTablePDF(FPDF):
    """论文级别表格PDF生成器"""

    def __init__(self):
        super().__init__(orientation='L', unit='mm', format='A4')
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        self.set_font('Helvetica', 'B', 16)
        self.set_text_color(*TEXT_COLOR)
        self.cell(0, 10, 'PhaseFlow Data Statistics', ln=True, align='C')
        self.set_font('Helvetica', '', 10)
        self.cell(0, 5, 'Manuscript Submission Tables', ln=True, align='C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def draw_table(self, title, headers, rows, col_widths, footnotes=None):
        """绘制专业表格"""
        self.add_page()
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(*TEXT_COLOR)
        self.cell(0, 8, title, ln=True, align='L')
        self.ln(3)

        # 计算表格尺寸
        total_width = sum(col_widths)
        start_x = (210 - total_width) / 2  # A4宽度210mm，居中

        # 绘制表头
        self.set_font('Helvetica', 'B', 9)
        self.set_fill_color(*HEADER_BG)
        self.set_text_color(*HEADER_TEXT)
        self.set_x(start_x)

        for i, (header, width) in enumerate(zip(headers, col_widths)):
            self.cell(width, 7, header, border=1, fill=True, align='C')
        self.ln()

        # 绘制数据行
        self.set_font('Helvetica', '', 8)
        self.set_text_color(*TEXT_COLOR)

        for row_idx, row in enumerate(rows):
            # 斑马纹背景
            if row_idx % 2 == 0:
                self.set_fill_color(*ALT_ROW_1)
            else:
                self.set_fill_color(*ALT_ROW_2)

            self.set_x(start_x)
            for cell, width in zip(row, col_widths):
                self.cell(width, 6, str(cell), border=1, fill=True, align='C')
            self.ln()

        # 绘制表注
        if footnotes:
            self.ln(3)
            self.set_font('Helvetica', 'I', 7)
            self.set_text_color(100, 100, 100)
            self.set_x(start_x)
            self.multi_cell(total_width, 4, footnotes, align='L')


# =============================================================================
# LLPS 表格数据
# =============================================================================

def get_llps_tables():
    """获取LLPS表格数据"""

    # 表1: 数据来源
    t1_headers = ['Data Source', 'Total', 'Hard Pos', 'Pseudo Pos', 'Struct Neg', 'Diso Neg']
    t1_rows = [
        ['UniProt/Swiss-Prot', '545,625', '0', '161', '0', '0'],
        ['RCSB PDB SEQRES', '199,311', '12', '18', '199,281', '0'],
        ['CD-CODE v2.2', '7,532', '121', '2,879', '676', '175'],
        ['DrLLPS', '1,651', '0', '716', '105', '12'],
        ['PhaSepDB 3.0', '1,119', '305', '807', '0', '0'],
        ['DisProt', '898', '0', '1', '0', '897'],
        ['BAV-LLPS DB', '4,435', '35', '2', '70', '14'],
        ['LLPSDB v2.0', '107', '52', '31', '7', '2'],
        ['CD-CODE', '42', '18', '18', '0', '0'],
    ]
    t1_widths = [50, 25, 22, 22, 25, 22]
    t1_footnote = "Abbreviations: Hard Pos, experimentally validated LLPS driver proteins; Pseudo Pos, weakly supervised positive samples; Struct Neg, structured negatives from PDB; Diso Neg, disordered negatives from DisProt."

    # 表2: 标注层级
    t2_headers = ['Tier', 'Type', 'Count', '%']
    t2_rows = [
        ['Gold', 'Hard Positive (experimental)', '543', '0.07%'],
        ['Gold', 'DPR Gold Span', '19 proteins', '<0.01%'],
        ['Silver', 'Pseudo Positive (weakly supervised)', '4,633', '0.61%'],
        ['Silver', 'DPR Silver Span', '141 spans', '<0.01%'],
        ['Negative', 'Structured Negative (PDB)', '200,139', '26.31%'],
        ['Negative', 'Disordered Negative (DisProt)', '1,100', '0.14%'],
        ['Unlabeled', 'PU Learning / Context', '554,305', '72.87%'],
    ]
    t2_widths = [30, 70, 35, 25]
    t2_footnote = "Gold tier contains experimentally validated data; Silver tier contains weakly supervised data."

    # 表3: 长度分布
    t3_headers = ['Length Range', 'Proteins', '%']
    t3_rows = [
        ['<30 aa (short peptide)', '28,061', '3.69%'],
        ['30-100 aa', '75,112', '9.87%'],
        ['100-2,048 aa', '653,782', '85.95%'],
        ['2,048-2,700 aa', '1,892', '0.25%'],
        ['2,700-5,537 aa', '1,721', '0.23%'],
        ['>5,537 aa', '152', '0.02%'],
    ]
    t3_widths = [60, 35, 25]
    t3_footnote = "The majority of proteins (85.95%) fall within the normal length range of 100-2,048 amino acids."

    # 表4: 物种分布
    t4_headers = ['Species (common)', 'Species (Latin)', 'Count', '%']
    t4_rows = [
        ['Human', 'Homo sapiens', '154', '28.4%'],
        ['Mouse', 'Mus musculus', '61', '11.2%'],
        ['Arabidopsis', 'Arabidopsis thaliana', '25', '4.6%'],
        ['Budding yeast', 'Saccharomyces cerevisiae', '23', '4.2%'],
        ['Reovirus', 'Reovirus type 1', '7', '1.3%'],
        ['Cow', 'Bos taurus', '6', '1.1%'],
        ['E. coli', 'Escherichia coli', '6', '1.1%'],
        ['Fission yeast', 'Schizosaccharomyces pombe', '5', '0.9%'],
        ['Mumps virus', 'Mumps virus', '5', '0.9%'],
        ['Other', 'Other', '251', '46.2%'],
    ]
    t4_widths = [35, 60, 25, 20]
    t4_footnote = "Human and mouse account for 39.6% of all experimentally validated LLPS driver proteins."

    # 表5: 数据引用
    t5_headers = ['Database', 'DOI / URL']
    t5_rows = [
        ['PPMC-lab LLPS Datasets', 'doi.org/10.5281/zenodo.15118996'],
        ['PhaSePro', 'doi.org/10.1093/nar/gkz848'],
        ['PhaSepDB 3.0', 'doi.org/10.1093/nar/gkz921'],
        ['LLPSDB v2.0', 'doi.org/10.1093/bioinformatics/btac026'],
        ['DrLLPS', 'doi.org/10.1093/nar/gkz1027'],
        ['CD-CODE', 'doi.org/10.1038/s41592-023-01831-0'],
        ['DisProt', 'doi.org/10.1093/nar/gkad928'],
        ['UniProtKB', 'doi.org/10.1093/nar/gkad945'],
    ]
    t5_widths = [55, 100]
    t5_footnote = "All training data sources are publicly available scientific databases."

    return [
        ('Table 1: LLPS Training Data Source Distribution', t1_headers, t1_rows, t1_widths, t1_footnote),
        ('Table 2: LLPS Label Tier Distribution', t2_headers, t2_rows, t2_widths, t2_footnote),
        ('Table 3: Sequence Length Distribution', t3_headers, t3_rows, t3_widths, t3_footnote),
        ('Table 4: Species Distribution of Hard Positive Proteins', t4_headers, t4_rows, t4_widths, t4_footnote),
        ('Table 5: Data Source Citations', t5_headers, t5_rows, t5_widths, t5_footnote),
    ]


# =============================================================================
# DPR 表格数据
# =============================================================================

def get_dpr_tables():
    """获取DPR表格数据"""

    # 表1: DPR训练池
    t1_headers = ['Training Tier', 'Label Tier', 'Proteins', 'Pos Residues', 'Spans']
    t1_rows = [
        ['gold_high', 'gold_positive', '19', '7,222', '33'],
        ['gold_high', 'pseudo_positive_high', '4', '355', '6'],
        ['pseudo_weak', 'pseudo_positive_weak', '71', '27,444', '141'],
        ['hard_negative', 'negative_curated_disordered', '123', '0', '0'],
        ['structured_negative', 'negative_curated_structured', '26,393', '0', '0'],
        ['bag_context', 'associated_context_unlabeled', '7,512', '0', '0'],
        ['ignored', 'unknown_pu_unlabeled', '39,088', '0', '0'],
    ]
    t1_widths = [30, 55, 25, 30, 20]
    t1_footnote = "Gold tier contains experimental DPR regions from PhaSePro; Silver tier contains weakly supervised spans."

    # 表2: DPR银标来源
    t2_headers = ['Source Database', 'Proteins', '%']
    t2_rows = [
        ['PhaSepDB 3.0', '406', '76.3%'],
        ['LLPSDB v2.0', '81', '15.2%'],
        ['CD-CODE', '21', '3.9%'],
        ['RCSB PDB SEQRES', '15', '2.8%'],
        ['UniProt/Swiss-Prot', '6', '1.1%'],
        ['Other', '3', '0.6%'],
    ]
    t2_widths = [55, 35, 25]
    t2_footnote = "A total of 532 proteins have DPR silver span annotations."

    # 表3: DPR银标LLPS标签
    t3_headers = ['LLPS Sampler Group', 'Proteins', '%']
    t3_rows = [
        ['pseudo_positive', '279', '52.4%'],
        ['hard_positive', '221', '41.5%'],
        ['unknown_pu', '18', '3.4%'],
        ['structured_negative', '8', '1.5%'],
        ['associated_context', '5', '0.9%'],
        ['disordered_negative', '1', '0.2%'],
    ]
    t3_widths = [55, 35, 25]
    t3_footnote = "Over 93% of DPR silver proteins have LLPS positive labels."

    # 表4: DPR基准对比
    t4_headers = ['Model', 'AUPRC', 'Spearman', 'Region Recall', 'F1@IoU0.25']
    t4_rows = [
        ['PSTP-Scan', '0.703', '0.254', '0.664', '0.427'],
        ['PSPHunter', '0.540', '0.041', '0.121', '0.076'],
        ['catGRANULE2', '0.489', '-0.110', '0.571', '0.457'],
        ['PhaseFlow', '0.622', '0.125', '0.479', '0.211'],
    ]
    t4_widths = [40, 28, 28, 35, 28]
    t4_footnote = "PhaseFlow achieves competitive AUPRC (0.622) and demonstrates the lowest false DPR rate on negative proteins."

    # 表5: DPR标签语义
    t5_headers = ['Label Type', 'Loss Function', 'Description']
    t5_rows = [
        ['gold_positive', 'BCE + Dice', 'PhaSePro experimental DPR regions'],
        ['pseudo_positive_high', 'BCE + Dice', 'High-confidence weakly supervised spans'],
        ['pseudo_positive_weak', 'BCE (low weight)', 'Low-confidence weakly supervised spans'],
        ['negative_curated', 'Negative loss', 'DisProt/PDB curated negatives'],
        ['associated_context', 'MIL only', 'Presence prediction only'],
        ['unknown_pu', 'Ignored', 'Not used in training'],
    ]
    t5_widths = [45, 35, 75]
    t5_footnote = "BCE, binary cross-entropy; MIL, multiple instance learning."

    # 表6: DPR训练配置
    t6_headers = ['Parameter', 'Value']
    t6_rows = [
        ['Model name', 'dpr_v3_portable_no_starling'],
        ['Training updates', '16,000'],
        ['PLM feature dimension', '1,280 (ESM2)'],
        ['Biophysical feature dimension', '112'],
        ['Maximum neighbors', '96'],
        ['Number of edge types', '10'],
        ['STARLING embedding', 'Disabled'],
        ['Protenix edges', 'Optional'],
    ]
    t6_widths = [55, 100]
    t6_footnote = "Checkpoint listed in the PhaseFlow paper artifact manifest."

    # 表7: PhaSePro基准
    t7_headers = ['Metric', 'Value']
    t7_rows = [
        ['Proteins', '121'],
        ['Total residues', '86,660'],
        ['Official regions', '143'],
        ['Positive residues', '46,288'],
        ['Protenix graph success', '121'],
        ['STARLING graph success', '0'],
    ]
    t7_widths = [55, 35]
    t7_footnote = "PhaSePro provides experimentally validated LLPS driver proteins and DPR/segment boundaries. DOI: doi.org/10.1093/nar/gkz848"

    return [
        ('Table S1: DPR Training Pool by Tier', t1_headers, t1_rows, t1_widths, t1_footnote),
        ('Table S2: DPR Silver Data Sources', t2_headers, t2_rows, t2_widths, t2_footnote),
        ('Table S3: DPR Silver LLPS Label Distribution', t3_headers, t3_rows, t3_widths, t3_footnote),
        ('Table S4: DPR Benchmark Comparison', t4_headers, t4_rows, t4_widths, t4_footnote),
        ('Table S5: DPR Label Semantics', t5_headers, t5_rows, t5_widths, t5_footnote),
        ('Table S6: DPR Training Configuration', t6_headers, t6_rows, t6_widths, t6_footnote),
        ('Table S7: PhaSePro Benchmark Dataset', t7_headers, t7_rows, t7_widths, t7_footnote),
    ]


# =============================================================================
# 主函数
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate PhaseFlow paper tables PDF')
    parser.add_argument('--task', type=str, choices=['llps', 'dpr', 'all'], default='all')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    if args.task in ['llps', 'all']:
        llps_dir = args.output_dir or DEFAULT_LLPS_DIR
        pdf = PaperTablePDF()
        for title, headers, rows, widths, footnote in get_llps_tables():
            pdf.draw_table(title, headers, rows, widths, footnote)
        output_path = Path(llps_dir) / 'tables.pdf'
        pdf.output(str(output_path))
        print(f'Generated: {output_path}')

    if args.task in ['dpr', 'all']:
        dpr_dir = args.output_dir or DEFAULT_DPR_DIR
        pdf = PaperTablePDF()
        for title, headers, rows, widths, footnote in get_dpr_tables():
            pdf.draw_table(title, headers, rows, widths, footnote)
        output_path = Path(dpr_dir) / 'tables.pdf'
        pdf.output(str(output_path))
        print(f'Generated: {output_path}')

    print('\nAll PDF tables generated successfully!')


if __name__ == '__main__':
    main()
