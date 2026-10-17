"""Build two documents with the official Statistica Sinica layout.
The downloaded official templates are preserved under official-template/.
Edit full-draft.tex and numerical-*.tex, then run this script.
"""
from pathlib import Path
root=Path(__file__).resolve().parents[1];s=(root/'full-draft.tex').read_text()
macros=s[s.index(r'\newtheorem{theorem}'):s.index(r'\date{}')]
preamble=r'''
\usepackage[sectionbib,authoryear,round]{natbib}
\usepackage{array,epsfig,fancyhdr,rotating}
\usepackage{amsmath,amssymb,amsfonts,amsthm,graphicx,booktabs,multirow,placeins}
\allowdisplaybreaks[1]
\usepackage{sectsty,secdot}
\usepackage{xr-hyper}
\usepackage[hidelinks]{hyperref}
\renewcommand{\bibsection}{\section*{References}}
\usepackage{etoolbox}
\AtBeginDocument{\apptocmd{\thebibliography}{\interlinepenalty=10000\clubpenalty=10000\widowpenalty=10000}{}{}}
\sectionfont{\fontsize{12}{14pt plus.8pt minus .6pt}\selectfont}
\subsectionfont{\fontsize{12}{14pt plus.8pt minus .6pt}\selectfont}
\textwidth=31.9pc
\textheight=46.5pc
\oddsidemargin=1pc
\evensidemargin=1pc
\headsep=15pt
\topmargin=.6cm
\parindent=1.7pc
\parskip=0pt
\emergencystretch=1em
\clubpenalty=10000
\widowpenalty=10000
\brokenpenalty=10000
\setcounter{page}{1}
\pagestyle{fancy}
\renewcommand{\sectionmark}[1]{}
\renewcommand{\subsectionmark}[1]{}
\lhead[\footnotesize\thepage\fancyplain{}\leftmark]{}
\chead[]{}
\rhead[]{\fancyplain{}\rightmark\footnotesize\thepage}
\cfoot{}
\fancypagestyle{firstpage}{\fancyhf{}\fancyhead[L]{\footnotesize\rm Statistica Sinica}\fancyhead[R]{\footnotesize\thepage}}
'''+macros+r'''
\numberwithin{equation}{section}
\newcommand{\bodyfont}{\fontsize{12}{14pt plus.8pt minus .6pt}\selectfont}
\newcommand{\referencefont}{\bibhang=1.7pc\bibsep=2pt\fontsize{9}{14pt plus.8pt minus .6pt}\selectfont}
'''
header=r'''
\begin{document}
\renewcommand{\baselinestretch}{2}
\bodyfont
\markboth{\hfill{\footnotesize\rm HAOZHEN SHU, ZHENHUA LIN AND YINAN LIN}\hfill}
{\hfill{\footnotesize\rm BANDS FOR FUNCTIONAL REGRESSION}\hfill}
\thispagestyle{firstpage}
$\ $\par\vspace{0.8pc}
\centerline{\large\bf SIMULTANEOUS CONFIDENCE BANDS FOR}
\vspace{2pt}
\centerline{\large\bf FUNCTIONAL VARYING-COEFFICIENT MODELS}
\vspace{.4cm}
\centerline{Haozhen Shu$^1$, Zhenhua Lin$^2$ and Yinan Lin$^3$}
\vspace{.4cm}
\centerline{\it $^1$National Institute of Education,}
\centerline{\it Nanyang Technological University,}
\centerline{\it $^2$National University of Singapore and}
\centerline{\it $^3$Chongqing Normal University}
\vspace{.55cm}
'''
abstract=s[s.index(r'\begin{abstract}')+len(r'\begin{abstract}'):s.index(r'\end{abstract}')]
body=s[s.index(r'\section{Introduction}'):s.index(r'\appendix')].strip()
if body.endswith(r'\clearpage'):body=body[:-len(r'\clearpage')]
appendix_start=s.index(r'\section{',s.index(r'\appendix'))
appendix=s[appendix_start:s.index(r'\bibliographystyle')].strip()
if appendix.endswith(r'\clearpage'):appendix=appendix[:-len(r'\clearpage')]
def clean(v):return '\n'.join(x for x in v.splitlines() if not x.lstrip().startswith('%'))
body=clean(body);appendix=clean(appendix)
refs=r'''\referencefont
\bibliographystyle{Chicago}
\bibliography{bibfile}
'''
back=r'''
\section*{Supplementary Material}
The supplementary document contains proofs, sampling-frequency and bandwidth results, the high-dimensional mean extension, and additional numerical and data-processing details. Reproduction code and complete simulation records accompany the manuscript.
'''+refs+r'''
\bigskip
\noindent\begin{minipage}{\textwidth}
\noindent Haozhen Shu. National Institute of Education, Nanyang Technological University, Singapore.\\
E-mail: \href{mailto:haozhen.shu@nie.edu.sg}{haozhen.shu@nie.edu.sg}
\medskip

\noindent Zhenhua Lin. Department of Statistics and Data Science, National University of Singapore, Singapore.\\
E-mail: \href{mailto:linz@nus.edu.sg}{linz@nus.edu.sg}
\medskip

\noindent Yinan Lin. National Center for Applied Mathematics in Chongqing, Chongqing Normal University, Chongqing, China.\\
E-mail: \href{mailto:linyn@cqnu.edu.cn}{linyn@cqnu.edu.cn}
\end{minipage}
\end{document}
'''
main=r'\documentclass[12pt]{article}'+'\n'+preamble+r'\externaldocument[][nocite]{fscb-supplement}'+'\n'+header+r'''
\fontsize{9}{11.5pt plus.8pt minus.6pt}\selectfont
\begin{quotation}
\noindent {\it Abstract:} '''+abstract.strip()+r'''
\par
\vspace{9pt}
\noindent {\it Key words and phrases:} Functional data, Gaussian approximation, multiplier bootstrap, simultaneous inference, varying-coefficient model.
\end{quotation}
\bodyfont
'''+body+back
supp=r'\documentclass[10pt]{book}'+'\n'+preamble.replace(r'\rm Statistica Sinica}',r'\rm Statistica Sinica: Supplement}')+r'\externaldocument[][nocite]{fscb-draft}'+'\n'+header+r'''
\raggedbottom
\centerline{\bf Supplementary Material}
\vspace{.55cm}
\fontsize{9}{11.5pt plus.8pt minus.6pt}\selectfont
This supplement uses the notation of the main article. Sections S1--S2 prove the main results under the assumptions stated in the main article. Section S3 discusses sampling frequency, dimension, bandwidth, and implementation. Section S4 extends the analysis to high-dimensional functional means. Section S5 reports computational details and additional results.
\par\bodyfont
\setcounter{section}{0}
\def\thesection{S\arabic{section}}
\numberwithin{equation}{section}
\numberwithin{lemma}{section}
\numberwithin{proposition}{section}
\numberwithin{corollary}{section}
\numberwithin{theorem}{section}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}
'''+appendix+'\n'+refs+r'\end{document}'
(root/'fscb-draft.tex').write_text(main);(root/'fscb-supplement.tex').write_text(supp)
