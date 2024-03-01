% Author: Andreu Gimenez Bolinches <esdandreu@gmail.com>
% Adapted from Felix Duvallet's RI thesis template:
% https://github.com/felixduvallet/ri-thesis-template
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{src/keio-master-thesis}

\LoadClassWithOptions{book}

%% Default packages setup -----------------------------------------------------
% Appendix
\RequirePackage{appendix}
% Colors
\RequirePackage{xcolor}  % Get extra colours.
\colorlet{documentLinkColor}{blue}
\colorlet{documentCitationColor}{black!80}
\definecolor{headergray}{rgb}{0.5,0.5,0.5}
% Hyperlinks
\RequirePackage[
    pageanchor=true,
    plainpages=false,
    pdfpagelabels,
    bookmarks,
    bookmarksnumbered,
]{hyperref}
\hypersetup{
    colorlinks = true,
    citecolor = documentCitationColor,
    linkcolor = documentLinkColor,
    urlcolor = documentLinkColor,
}
% Figures
\RequirePackage{graphicx}
% Fancy headers
\RequirePackage{fancyhdr}

%% Header styling -------------------------------------------------------------

% Remove rulers around the chapter titles
\renewcommand{\headrulewidth}{0.0pt}
\renewcommand{\footrulewidth}{0.0pt}

% headers
\fancyhead[LO,R]{\hyperlink{contents}{\slshape \leftmark}}
\fancyhead[LO,R]{\hyperlink{contents}{\slshape \leftmark}}
% Make chapter header line: N. <name>
\renewcommand{\chaptermark}[1]{%
    \markboth{%
        \color{headergray}{%
            \thechapter.\ #1%
        }%
    }{}%
}

% footers
\fancyfoot[C]{}

%% Page styles ----------------------------------------------------------------

\fancypagestyle{plain}{%
    \fancyhf{}
    \fancyfoot[RO,LE]{\thepage}
}

\fancypagestyle{thesis}{%
    \fancyhead{}
    \fancyhead[RO,LE]{\hyperlink{contents}{\slshape \leftmark}}
    \fancyfoot[RO,LE]{\thepage}%
}

\fancypagestyle{title}{%
    \fancyhead[R]{Academic Year \@ifundefined{@Year}{}{\@Year}}
    \fancyhead[L]{Master's Thesis}
    \fancyfoot{}
}

\setlength{\headheight}{15pt}
\addtolength{\topmargin}{-3pt}

%% Title ----------------------------------------------------------------------

% Parameters:
% Required
\def\title#1{\gdef\@title{#1}}
\def\author#1{\gdef\@author{#1}}
\def\authorID#1{\gdef\@authorID{#1}}
\def\advisora#1{\gdef\@advisora{#1}}
\def\advisorb#1{\gdef\@advisorb{#1}}
\def\school#1{\gdef\@school{#1}}
\def\subschool#1{\gdef\@subschool{#1}}
\def\date#1{\gdef\@date{#1}}
\def\Year#1{\gdef\@Year{#1}}
\def\MonthName#1{%
  \ifcase #1
  \or January% 1
  \or February% 2
  \or March% 3
  \or April% 4
  \or May% 5
  \or June% 6
  \or July% 7
  \or August% 8
  \or September% 9
  \or October% 10
  \or November% 11
  \or December% 12
  \fi}
\def\Month#1{\gdef\@Month{#1}}

\def\maketitle{
    \thispagestyle{title}
    
    % calculate skip needed to ensure that title appears in the cut-out
    \newlength{\@cutoutvskip}
    \setlength{\@cutoutvskip}{2.1875 true in}       % position of cut-out
    \addtolength{\@cutoutvskip}{-1 true in}         % printer margin
    \addtolength{\@cutoutvskip}{-\topmargin}
    \addtolength{\@cutoutvskip}{-\headheight}
    \addtolength{\@cutoutvskip}{-\headsep}
    
    %% Centered things on the title page must be *physically* centered
    %% on the page, so they line up with the cut-out window. So we hardwire
    %% the margins for the title page so that left margin = right margin:
    %%         left margin = (8.5in - textwidth)/2
    \oddsidemargin=8.5in
    \advance\oddsidemargin by -\textwidth
    \oddsidemargin=.5\oddsidemargin
    \advance\oddsidemargin by -1in % TeX sux
    \let\footnoterule\relax
    \vglue\@cutoutvskip
    
    % Keio University format for master’s thesis title page academic year 2022
    
    % Thesis title
    \begin{center}
        \begin{minipage}[t]{.8\textwidth}
            \vfill
            \begin{center}
                {\Huge \strut \bf \@title \par}
            \end{center}
            \vfill
        \end{minipage}
    \end{center}
    
    % Author
    \vfill
    \begin{center}
        {\LARGE \bf \@author \par} \medskip
        {\large ({\bf Student ID No.:} \@authorID)}
    \end{center}
    \vspace{1em}
    
    % Advisor
    \vfill
    \begin{center}
        \begin{tabular}{cl}
            \Large {\bf Advisor} & \Large \@advisora \\
                                 & \Large \@advisorb
        \end{tabular}
    \end{center}
    
    % Date
    \vfill
    \begin{center}
        {\large \MonthName{\@Month} \@Year \par}
    \end{center}
    
    % School
    \vfill
    \begin{center}
        \bf Keio University \par
        \@school \par
        School of \@subschool \par
    \end{center}
    \clearpage
}

%% Preface

\newenvironment{dedication}
{
    \thispagestyle{fancy}
    \vspace*{\stretch{1}} \begin{center} \em
        }
        {
    \end{center} \vspace*{\stretch{3}} \clearpage
}

\newenvironment{pseudochapter}[1]{
    \thispagestyle{fancy}
    %%\vspace*{\stretch{1}}
    \begin{center} \large {\bf #1} \end{center}
    \begin{quotation}
    }{
    \end{quotation}
    \vspace*{\stretch{3}}
    \clearpage
}

\newenvironment{abstract}{
    \begin{pseudochapter}{Thesis Abstract}}{\end{pseudochapter}
}

\newenvironment{acknowledgments}{
    \begin{pseudochapter}{Acknowledgments}}{\end{pseudochapter}
}

\newenvironment{funding}{
    \begin{pseudochapter}{Funding}}{\end{pseudochapter}
}
