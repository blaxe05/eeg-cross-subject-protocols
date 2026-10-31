"""Create source-backed architecture and experiment schematics as HTML/SVG.

The self-contained HTML is the editable source. Optional PDF/PNG rendering can
use a browser; no generated data, EEG traces, or result values are drawn.
"""
from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
import re


INK, MUTED, RULE = '#202B33', '#52616D', '#7B8790'
BLUE, TEAL, ORANGE = '#245A81', '#00796B', '#A6532F'


def text(x, y, value, size=18, weight='normal', color=INK, anchor='middle'):
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{color}" font-size="{size}" font-weight="{weight}" font-family="Arial,Helvetica,sans-serif">{escape(value)}</text>'


def box(x, y, w, h, lines, stroke=RULE, fill='white', key=''):
    tag = f' data-node-id="{key}"' if key else ''
    parts = [f'<g{tag}><rect data-role="bounds" x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>']
    first_y = y + h / 2 - (len(lines) - 1) * 12 + 6
    for i, line in enumerate(lines):
        parts.append(text(x + w / 2, first_y + i * 24, line, 18 if i == 0 else 16.5,
                          'bold' if i == 0 else 'normal'))
    return ''.join(parts) + '</g>'


def arrow(x1, y1, x2, y2, color=MUTED, dashed=False):
    assert x1 == x2 or y1 == y2, 'Connectors must remain orthogonal'
    dash = ' stroke-dasharray="7,5"' if dashed else ''
    return f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{color}" stroke-width="2"{dash} marker-end="url(#arrow)"/>'


def canvas(height, body, title, description):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="{height}" viewBox="0 0 1024 {height}" role="img" aria-labelledby="title desc">
<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>
<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto-start-reverse"><path d="M0,0 L8,4 L0,8 Z" fill="{MUTED}"/></marker></defs>
<rect width="1024" height="{height}" fill="white"/>{''.join(body)}</svg>'''


def architecture():
    parts = [text(24, 28, 'a   DGCNN', 20, 'bold', anchor='start'),
             '<rect x="112" y="48" width="704" height="248" rx="6" fill="#F5F8FB" stroke="#245A81" stroke-width="1.5"/>',
             text(128, 76, 'Learned-graph encoder', 18, 'bold', anchor='start'),
             arrow(312, 124, 360, 124), arrow(472, 124, 520, 124),
             arrow(96, 240, 184, 240), arrow(352, 240, 400, 240),
             arrow(512, 240, 568, 240), arrow(712, 240, 856, 240),
             '<path d="M616,152 V168 Q616,176 608,176 H276 Q268,176 268,184 V200" fill="none" stroke="#52616D" stroke-width="2" marker-end="url(#arrow)"/>',
             text(632, 176, 'L', 17, color=MUTED, anchor='start'),
             text(780, 228, 'C × F', 17),
             box(184, 96, 128, 56, ['A + b', 'learned'], key='learned_adjacency'),
             box(360, 96, 112, 56, ['ReLU'], key='adjacency_relu'),
             box(520, 96, 192, 56, ['Laplacian L', 'normalized'], key='graph_laplacian'),
             box(24, 200, 72, 80, ['Input', 'C × 5'], key='graph_input'),
             box(184, 200, 168, 80, ['GraphConv', 'K = 2', '[X, LX]W'], key='graph_filter'),
             box(400, 200, 112, 80, ['Dropout', 'p = 0.5'], key='graph_dropout'),
             box(568, 200, 144, 80, ['B1ReLU', 'ReLU(H + β)'], key='graph_activation'),
             box(856, 188, 152, 104, ['Classifier', 'flatten → FC256', 'FC → class logits', '2 / 3 / 4 / 9'], key='graph_classifier'),
             text(24, 328, 'F = 64 for C = 62 or 14; F = 128 for C = 32. Dropout precedes both classifier linear layers.', 17, color=MUTED, anchor='start'),
             text(24, 380, 'b   CDCN', 20, 'bold', anchor='start'),
             '<rect x="112" y="400" width="704" height="200" rx="6" fill="#FBF4F0" stroke="#A6532F" stroke-width="1.5"/>',
             text(128, 428, 'Channel-fused dense encoder', 18, 'bold', anchor='start')]
    for a, b in [(96, 144), (224, 248), (352, 376), (432, 456), (560, 584), (640, 664), (768, 856)]:
        parts.append(arrow(a, 500, b, 500))
    parts += [box(24, 460, 72, 80, ['Input', 'C × 5'], key='dense_input'),
              box(144, 460, 80, 80, ['Stem', '1 × 5', '24 maps'], key='dense_stem'),
              box(248, 460, 104, 80, ['DB1', '6 layers', '96 maps'], key='dense_block_1'),
              box(376, 460, 56, 80, ['T1'], key='transition_1'),
              box(456, 460, 104, 80, ['DB2', '6 layers', '168 maps'], key='dense_block_2'),
              box(584, 460, 56, 80, ['T2'], key='transition_2'),
              box(664, 460, 104, 80, ['DB3', '6 layers', '240 maps'], key='dense_block_3'),
              box(856, 444, 152, 112, ['Classifier', 'BN → ReLU', 'GAP → dropout', 'FC → 2/3/4/9'], key='dense_classifier'),
              text(184, 572, 'C × 1', 16.5), text(300, 572, 'C × 1', 16.5),
              text(508, 572, 'H₁ × 1', 16.5), text(716, 572, 'H₂ × 1', 16.5),
              text(24, 636, 'DB: six dense units, growth 12. Each unit: BN → ReLU → 3 × 1 conv → dropout → concatenate.', 17, anchor='start'),
              text(24, 668, 'T1 / T2: BN → ReLU → 1 × 1 conv → 2 × 1 max pool. T2 pads one channel row before pooling.', 17, anchor='start'),
              text(24, 700, 'H₁ = floor(C / 2); H₂ = floor((H₁ + 1) / 2). C = 62: SEED / SEED-IV; 32: FACED; 14: DREAMER.', 17, color=MUTED, anchor='start')]
    return canvas(724, parts, 'Layer architectures of the fixed DGCNN and CDCN implementations',
                  'The graph encoder shows the learned adjacency-to-Laplacian branch and K=2 filtering. The dense encoder shows its stem, three six-unit dense blocks, two transitions, feature-map counts, and spatial reductions. Each dense unit concatenates its new maps with all input maps. Classifier dropout is grouped in the head for readability.')


def design():
    parts = [text(24, 28, 'a   Checkpoint transfer  ·  unchanged fitted trajectory', 20, 'bold', anchor='start')]
    for a, b in [(212, 260), (472, 520), (752, 800)]:
        parts.append(arrow(a, 96, b, 96))
    parts += [box(24, 56, 188, 80, ['Source training', 'source labels'], key='source_train'),
              box(260, 56, 212, 80, ['Saved checkpoints', '40; FACED uses 8'], key='saved_states'),
              box(520, 56, 232, 80, ['Source validation', 'earliest BAcc maximum'], BLUE, '#F2F6FA', 'validation_selector'),
              box(800, 56, 200, 80, ['Different target', 'chosen versus final'], key='target_score'),
              text(24, 172, 'Selection-subject gain and target change use the same two model states; target labels are for scoring.', 17, anchor='start'),
              '<path d="M24,196 H1000" stroke="#D9E0E5" stroke-width="1"/>',
              text(24, 232, 'b   Matched source-training comparisons  ·  fixed target and initialization', 20, 'bold', anchor='start')]
    for a, b, label in [(228, 284, 'compare'), (488, 544, 'compare')]:
        parts.append(f'<path d="M{a},320 H{b}" fill="none" stroke="{MUTED}" stroke-width="1.5" stroke-dasharray="4,4" marker-start="url(#arrow)" marker-end="url(#arrow)"/>')
        parts.append(text((a+b)/2, 274, label, 16, color=MUTED))
    parts.append(arrow(748, 320, 804, 320))
    parts.append(text(776, 274, 'continue', 16, color=MUTED))
    parts += [box(24, 284, 204, 80, ['A  Original sources', '11 subjects · N · S'], key='arm_a'),
              box(284, 284, 204, 80, ['B  New composition', '14 subjects · N · S'], key='arm_b'),
              box(544, 284, 204, 80, ['C  More examples', '14 subjects · Nₑ · S'], key='arm_c'),
              box(804, 284, 204, 80, ['D  More updates', 'same Nₑ · S′ > S'], key='arm_d'),
              text(24, 400, 'A / B / C share initialization. B matches A’s class counts. C contains B’s examples; D continues C.', 17, anchor='start'),
              '<path d="M24,424 H1000" stroke="#D9E0E5" stroke-width="1"/>',
              text(24, 460, 'c   Auxiliary-input controls  ·  same backbone, source batches and update budget', 20, 'bold', anchor='start'),
              box(24, 488, 288, 80, ['Source proxy · loss off', 'domain coefficient λ = 0'], key='adapt_off'),
              box(368, 488, 288, 80, ['Source proxy · loss on', 'domain coefficient λ = 1'], key='adapt_proxy'),
              box(712, 488, 288, 80, ['Target EEG · loss on', 'same λ = 1; no emotion labels'], ORANGE, '#FBF4F0', 'adapt_target'),
              text(24, 604, 'Compare arms at the fixed final step. CDCN auxiliary passes use frozen source-derived BatchNorm statistics.', 17, anchor='start')]
    return canvas(628, parts, 'Information paths and matched experimental contrasts',
                  'Three panels distinguish source-only checkpoint selection and target scoring, a source-training path, and parallel auxiliary-input adaptation controls. The source path shows an ordered comparison, not sequential fitting of all four arms; only C and D share fitted optimizer history.')


def partition_budget():
    """Reference fold counts and feature-model schedules; no outcome data."""
    roles = [('Training · fit parameters', BLUE, '#EAF1F7'),
             ('Validation · select checkpoint', ORANGE, '#FBF0E7'),
             ('Target · score only', MUTED, '#52616D')]
    parts = [text(24, 28, 'a   Reference subject partitions', 20, 'bold', anchor='start')]
    for x, (label, stroke, fill) in zip([24, 352, 744], roles):
        parts.append(f'<rect x="{x}" y="54" width="16" height="16" fill="{fill}" stroke="{stroke}" stroke-width="1.25"/>')
        parts.append(text(x+26, 68, label, 17, anchor='start'))
    specs = [('SEED / SEED-IV', 'each n = 15', (11, 3, 1), 100),
             ('FACED', 'n = 123', (110, 12, 1), 164),
             ('DREAMER V / A', 'shared n = 23', (17, 5, 1), 228)]
    for label, total_label, counts, y in specs:
        parts += [text(24, y+18, label, 18, 'bold', anchor='start'),
                  text(24, y+41, total_label, 16.5, color=MUTED, anchor='start')]
        total = sum(counts);x = 260
        for role, (count, (_, stroke, fill)) in enumerate(zip(counts, roles)):
            width = 704*count/total
            parts.append(f'<rect data-role-index="{role}" data-subject-count="{count}" x="{x:.4f}" y="{y}" width="{width:.4f}" height="40" fill="{fill}" stroke="{stroke}" stroke-width="1.25"/>')
            if width >= 24:
                parts.append(text(x+width/2, y+27, str(count), 19, 'bold', color='white' if role==2 else stroke))
            else:
                parts.append(f'<path d="M{x+width/2:.4f},{y+20} H982" stroke="{MUTED}" stroke-width="1.25" fill="none"/>')
                parts.append(text(998, y+27, str(count), 19, 'bold'))
            x += width
    parts += [text(24, 298, 'Roles are disjoint within each reference fold; every subject is evaluated as the outer target.', 17, anchor='start'),
              '<path d="M24,322 H1000" stroke="#D9E0E5" stroke-width="1"/>',
              text(24, 358, 'b   Update budgets and saved checkpoints', 20, 'bold', anchor='start'),
              text(24, 400, 'Sₜ = E × ⌈Nₜ / 256⌉', 23, 'bold', anchor='start'),
              text(544, 394, 'Checkpoint positions / own update budget', 17, anchor='start'),
              text(24, 430, 'Task', 17, 'bold', anchor='start'),
              text(300, 430, 'Nₜ', 17, 'bold', anchor='end'),
              text(360, 430, 'E', 17, 'bold', anchor='end'),
              text(476, 430, 'Updates Sₜ', 17, 'bold', anchor='end'),
              '<path d="M24,438 H476" stroke="#7B8790" stroke-width="1"/>']
    budgets = [('SEED', 9416, 40, 1480), ('SEED-IV', 7139, 40, 1120),
               ('FACED', 30800, 8, 968), ('DREAMER V/A', 3060, 40, 480)]
    for y, (label, anchors, units, updates) in zip([464, 496, 528, 560], budgets):
        assert updates == units*((anchors+255)//256)
        parts += [text(24, y, label, 17.5, anchor='start'),
                  text(300, y, f'{anchors:,}', 17.5, anchor='end'),
                  text(360, y, str(units), 17.5, anchor='end'),
                  text(476, y, f'{updates:,}', 17.5, anchor='end')]
    for count, y in [(40, 458), (8, 536)]:
        label = '40 checkpoints · SEED / SEED-IV / DREAMER' if count==40 else '8 checkpoints · FACED'
        parts.append(text(544, y-25, label, 17, anchor='start'))
        parts.append(f'<path d="M560,{y} H964" stroke="#7B8790" stroke-width="1.25"/>')
        for k in range(1, count+1):
            x = 560+404*k/count
            parts.append(f'<circle data-checkpoint-count="{count}" data-checkpoint="{k}" data-fraction="{k/count:.8f}" cx="{x:.4f}" cy="{y}" r="{5 if k==count else 2.7}" fill="{INK if k==count else BLUE}"/>')
        parts += [text(560, y+25, '0', 16, color=MUTED),
                  text(964, y+25, 'Sₜ', 16, color=MUTED)]
    parts += [text(24, 600, 'Nₜ counts training anchors; each update processes 256 examples. The final checkpoint is included.', 17, anchor='start')]
    svg = canvas(624, parts, 'Reference subject partitions, optimizer-update budgets and checkpoint schedules',
                 'Partition segments are proportional within each dataset. Counts are training, reserved source validation and target subjects: 11/3/1 for SEED and SEED-IV, 110/12/1 for FACED and 17/5/1 for both DREAMER tasks. Reference update counts are 1480, 1120, 968 and 480. Forty or eight saved checkpoints span a schedule normalized to its own final update; no initial checkpoint is selected. These are feature-based reference schedules, not the temporal-model schedules.')
    return svg.replace('id="title"', 'id="partition-budget-title"').replace('id="desc"', 'id="partition-budget-desc"').replace('aria-labelledby="title desc"', 'aria-labelledby="partition-budget-title partition-budget-desc"')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, svg in [('backbone_architectures', architecture()), ('controlled_experiment_design', design()), ('subject_partitions_budgets', partition_budget())]:
        height = re.search(r'height="(\d+)"', svg)[1]
        html = ('<!doctype html><html><head><meta charset="utf-8"><style>'
                f'@page{{size:7.12in {int(height)/1024*7.12:.5f}in;margin:0}}'
                'html,body{margin:0;background:white}svg{display:block;width:100%;height:auto}'
                '</style></head><body>' + svg + '</body></html>')
        (args.output_dir / f'{name}.html').write_text(html, encoding='utf8')
        # Export from the authored HTML, preserving the inline SVG as source.
        inline = re.search(r'(<svg\b.*?</svg>)', html, re.S)[1]
        (args.output_dir / f'{name}.svg').write_text(inline, encoding='utf8')


if __name__ == '__main__':
    main()
