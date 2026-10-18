"""Compact, data-preserving simulation tables without visual rankings.

All numerical inputs are read unchanged from validated summary rows. Coverage
is displayed as a percentage to one decimal; average width has three decimals.
Method comparisons use paired entries. High-dimensional results show only
the proposed method, with paired p50/100 entries on six sampling rows.

For a formatting-only preview of an already validated summary:
  python code/simulation_table_format.py --output-directory results/table-format-preview
The authoritative reporter retains all original data/provenance gates before
using these same rendering functions for --update-tables.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
VERSION='unranked-method-pairs-compact-dimensions-v4'
NS,MS=(200,400),(5,20,80)
LAWS=('normal','laplace','chisq5')
NAMES={1:'mean.tex',2:'single.tex',5:'joint.tex',100:'highd-mean.tex'}


def number(row,name):
    value=float(row[name])
    if not math.isfinite(value):raise ValueError('Nonfinite table input: '+name)
    return value


def integer(row,name):
    value=number(row,name)
    if not value.is_integer():raise ValueError('Noninteger table metadata: '+name)
    return int(value)


def validate_row(row):
    if (integer(row,'R')!=1000 or integer(row,'degree')!=1
            or integer(row,'grid_size')!=51 or number(row,'grid_start')!=.15
            or number(row,'grid_end')!=.85):
        raise ValueError('Unexpected simulation table R/degree/evaluation grid.')


def metrics(row,rule):
    cp,aw=number(row,rule+'_CP'),number(row,rule+'_AW')
    if not 0<=cp<=1 or aw<=0:raise ValueError('Invalid table coverage/width.')
    if abs(cp*1000-round(cp*1000))>1e-8:
        raise ValueError('Coverage does not correspond to 1000 retained replications.')
    return cp,aw


def pair(left,right,digits=1):
    return f'{left:.{digits}f}'+r'\,/\,'+f'{right:.{digits}f}'


def coverage_pair(left,right):
    return pair(100*left,100*right)


def begin(caption,label,dimension_rows=False):
    common=[r'\begin{table}[!tbp]',r'\centering',r'\caption{'+caption+'}',
            r'\label{'+label+'}',r'\renewcommand{\baselinestretch}{1}\footnotesize',
            r'\setlength{\tabcolsep}{3pt}',r'\renewcommand{\arraystretch}{1.08}',
            ]
    if dimension_rows:
        header=[r'\begin{tabular}{@{}rrr*{3}{cc}@{}}',r'\toprule',
                r'& & & \multicolumn{2}{c}{Gaussian} & \multicolumn{2}{c}{Laplace} & \multicolumn{2}{c}{Adjusted $\chi^2_5$}\\',
                r'\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}',
                r'$n$ & $m$ & $p$ & CP (\%) & AW & CP (\%) & AW & CP (\%) & AW\\']
    else:
        header=[r'\begin{tabular}{@{}rr*{3}{cc}@{}}',r'\toprule',
                r'& & \multicolumn{2}{c}{Gaussian} & \multicolumn{2}{c}{Laplace} & \multicolumn{2}{c}{Adjusted $\chi^2_5$}\\',
                r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}',
                r'$n$ & $m$ & CP (\%) & AW & CP (\%) & AW & CP (\%) & AW\\']
    return common+header+[r'\midrule']


def finish(lines):
    return '\n'.join(lines+[r'\bottomrule',r'\end{tabular}',r'\end{table}'])+'\n'


def comparison_table(rows,p):
    if p not in (1,2,5):raise ValueError('Unknown method-comparison target.')
    labels={1:'tab:mean-new',2:'tab:single-new',5:'tab:joint-new'}
    objects={1:'a scalar functional mean',2:r'$\beta_2$ in the two-coefficient model',
             5:'all five coefficients jointly'}
    comparator='Cai--Hu' if p==1 else 'CLZ'
    selected=[row for row in rows if integer(row,'target_p')==p]
    lookup={}
    for row in selected:
        validate_row(row)
        k=integer(row,'n'),integer(row,'m'),row['law']
        if k in lookup:raise ValueError('Duplicate method-comparison table cell.')
        lookup[k]=row
    expected={(n,m,law) for n in NS for m in MS for law in LAWS}
    if set(lookup)!=expected:raise ValueError('Incomplete method-comparison table.')
    caption={1:r'Scalar mean bands: CP (\%) and AW (Proposed\,/\,Cai--Hu).',
             2:r'CP (\%) and AW for $\beta_2$ (Proposed\,/\,CLZ).',
             5:r'Joint coefficient bands: CP (\%) and AW (Proposed\,/\,Bonferroni CLZ).'}[p]
    lines=begin(caption,labels[p])
    for n in NS:
        if n!=NS[0]:lines.append(r'\addlinespace[3pt]')
        for m in MS:
            cells=[]
            for law in LAWS:
                row=lookup[n,m,law];rule=row['paper_rule']
                proposed_cp,proposed_aw=metrics(row,rule)
                comparator_cp,comparator_aw=metrics(row,'comparator')
                cells.extend([coverage_pair(proposed_cp,comparator_cp),
                              pair(proposed_aw,comparator_aw,digits=3)])
            lines.append(' & '.join([str(n) if m==MS[0] else '',str(m)]+cells)+r'\\')
    return finish(lines)


def highd_table(rows):
    lookup={}
    for row in rows:
        if integer(row,'target_p')!=100:continue
        validate_row(row)
        k=integer(row,'n'),integer(row,'m'),integer(row,'p'),row['law']
        if k in lookup:raise ValueError('Duplicate high-dimensional table cell.')
        lookup[k]=row
    expected={(n,m,p,law) for n in NS for m in MS for p in (50,100) for law in LAWS}
    if set(lookup)!=expected:raise ValueError('Incomplete high-dimensional table.')
    caption=r'Joint mean bands: CP (\%) and AW. Each pair lists $p=50/100$.'
    lines=begin(caption,'tab:highd-mean')
    for n in NS:
        if n!=NS[0]:lines.append(r'\addlinespace[3pt]')
        for m in MS:
            cells=[]
            for law in LAWS:
                left,right=[lookup[n,m,p,law] for p in (50,100)]
                lcp,lawidth=metrics(left,left['paper_rule'])
                rcp,rawidth=metrics(right,right['paper_rule'])
                cells.extend([coverage_pair(lcp,rcp),pair(lawidth,rawidth,digits=3)])
            lines.append(' & '.join([str(n) if m==MS[0] else '',str(m)]+cells)+r'\\')
    return finish(lines)


def tables(rows):
    if len(rows)!=90:raise ValueError('Formatting requires all 90 validated summary cells.')
    return {NAMES[p]:highd_table(rows) if p==100 else comparison_table(rows,p) for p in (2,5,1,100)}


def result_fragments(rows):
    """Qualitative interpretation of the unchanged finite-sample study."""
    # Verify the descriptive claims against all configurations before writing.
    single=[r for r in rows if integer(r,'target_p')==2]
    sparse=[r for r in single if integer(r,'m')==5]
    if len(sparse)!=6 or not all(abs(number(r,r['paper_rule']+'_CP')-.95)<
                                abs(number(r,'comparator_CP')-.95) for r in sparse):
        raise ValueError('The sparse single-coefficient interpretation requires review.')
    coefficients=[r for r in rows if integer(r,'target_p') in (2,5)]
    lookup={(integer(r,'target_p'),integer(r,'n'),integer(r,'m'),r['law']):r for r in coefficients}
    if not all(number(lookup[p,n,5,law],'comparator_CP')>number(lookup[p,n,20,law],'comparator_CP')>
               number(lookup[p,n,80,law],'comparator_CP') and
               number(lookup[p,n,5,law],'comparator_AW')>number(lookup[p,n,20,law],'comparator_AW')>
               number(lookup[p,n,80,law],'comparator_AW') for p in (2,5) for n in NS for law in LAWS):
        raise ValueError('The CLZ sampling-frequency interpretation requires review.')
    comparison=(
        'Under sparse sampling, the proposed single-coefficient bands are closer to nominal coverage than CLZ, although undercoverage remains. '
        "CLZ's leading sparse covariance emphasizes same-observation contributions. With persistent subject-level errors, distinct-visit covariance can still affect finite-sample variance at the bandwidths used here, even in sparse configurations. "
        'Prediction-based bandwidth selection can also leave smoothing bias; the coverage results do not isolate these contributions.\n\n'
        'As sampling becomes denser, CLZ bands narrow and their undercoverage worsens, consistent with limitations of the sparse covariance approximation. '
        'For the five-function target, Bonferroni addresses multiplicity but cannot repair undercoverage of the input marginal bands. '
        'The proposed joint bands also retain finite-sample undercoverage, especially in the smaller, sparse configurations.\n')
    gaussian_pairs=[(lookup[5,n,20,'normal'],lookup[5,n,80,'normal']) for n in NS]
    if not all(number(y,'selected_CP')>number(x,'selected_CP') for x,y in gaussian_pairs):
        raise ValueError('The Gaussian joint-band coverage interpretation requires review.')
    width_changes=[number(y,'selected_AW')-number(x,'selected_AW') for x,y in gaussian_pairs]
    if not min(width_changes)<0<max(width_changes):
        raise ValueError('The Gaussian joint-band width interpretation requires review.')
    gaussian=(
        'For Gaussian joint bands, coverage improves between the two denser sampling configurations, while width changes in different directions at the two sample sizes. '
        'These comparisons use the prespecified bandwidth rule and do not isolate the effect of additional observations.\n')
    scalar=[r for r in rows if integer(r,'target_p')==1]
    if len(scalar)!=18 or not all(number(r,'comparator_AW')<number(r,r['paper_rule']+'_AW') for r in scalar):
        raise ValueError('The scalar-mean width interpretation requires review.')
    scalar_text=(
        'The scalar-mean results are broadly comparable for the proposed and Cai--Hu bands on the same datasets and grid. '
        'Cai--Hu bands are narrower throughout this design, but both procedures show undercoverage in some configurations.\n')
    highd_text=(
        'The high-dimensional mean experiment illustrates joint inference for distinct mean curves under a shared scalar functional error. '
        'Some configurations still show undercoverage. These results concern a strongly dependent design and do not constitute a direct comparison with a native high-dimensional joint-band method.\n')
    return {'comparison-text.tex':comparison,'gaussian-rerun-text.tex':gaussian,
            'scalar-mean-text.tex':scalar_text,'highd-mean-text.tex':highd_text}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',type=Path,default=ROOT/'results/linear-only-20261018/final-paper-summary.csv')
    parser.add_argument('--output-directory',type=Path,required=True)
    args=parser.parse_args()
    if args.output_directory.resolve()==(ROOT/'results/tables').resolve():
        raise ValueError('Use the authoritative reporter --update-tables for paper replacement.')
    with args.summary.open(newline='') as stream:rows=list(csv.DictReader(stream))
    rendered={**tables(rows),**result_fragments(rows)};args.output_directory.mkdir(parents=True,exist_ok=True)
    for name,text in rendered.items():(args.output_directory/name).write_text(text)
    metadata=dict(formatter_version=VERSION,formatting_only_preview=True,
                  summary_path=str(args.summary.resolve()),summary_sha256=hashlib.sha256(args.summary.read_bytes()).hexdigest(),
                  formatter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  tables={name:hashlib.sha256(text.encode()).hexdigest() for name,text in rendered.items()},
                  coverage_display='100 times original probability; one decimal',
                  width_display='Original width; three decimals',
                  numerical_values_have_no_emphasis=True,
                  highd_proposed_only_with_separate_dimension_rows=True,
                  table_dimensions={name:dict(body_rows=12 if name=='highd-mean.tex' else 6,columns=9 if name=='highd-mean.tex' else 8) for name in NAMES.values()})
    (args.output_directory/'formatting-preview.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Formatting-only preview:',args.output_directory,'; paper tables unchanged.')


if __name__=='__main__':main()
