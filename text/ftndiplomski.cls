%%
%% ftndiplomski.cls
%% Unofficial LaTeX class for bachelor/master theses at the
%% Faculty of Technical Sciences, University of Novi Sad.
%%
%%
%% Compiles with XeLaTeX (required for fontspec/polyglossia and, if the
%% "serbian" language option is used, Cyrillic text).
%%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ftndiplomski}[2026/08/13 v1.1 FTN Novi Sad bachelor/master thesis]

\newif\if@ftn@twoside \@ftn@twosidefalse
\newif\if@ftn@onehalf \@ftn@onehalffalse
\newif\if@ftn@latin   \@ftn@latinfalse
\newif\if@ftn@english \@ftn@englishtrue

\DeclareOption{twoside}{\@ftn@twosidetrue}
\DeclareOption{oneside}{\@ftn@twosidefalse}
\DeclareOption{onehalfspacing}{\@ftn@onehalftrue}
% Thesis written in English (default).
\DeclareOption{english}{\@ftn@englishtrue}
% Thesis written in Serbian, Latin script.
\DeclareOption{serbianlatin}{\@ftn@englishfalse\@ftn@latintrue}
% Thesis written in Serbian, Cyrillic script.
\DeclareOption{serbian}{\@ftn@englishfalse\@ftn@latinfalse}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ProcessOptions\relax

\if@ftn@twoside
  \LoadClass[12pt,a4paper,twoside,openright]{report}
\else
  \LoadClass[12pt,a4paper,oneside]{report}
\fi

%% ------------------------------------------------------------------
%%  Stranica / geometrija
%% ------------------------------------------------------------------
\RequirePackage[a4paper,left=3cm,right=2.5cm,top=2.5cm,bottom=2.5cm,%
  headheight=15pt,headsep=8mm]{geometry}

%% ------------------------------------------------------------------
%%  Fontovi i jezik (XeLaTeX)
%% ------------------------------------------------------------------
\RequirePackage{fontspec}
\RequirePackage{polyglossia}

\if@ftn@english
  \setmainlanguage{english}
  \setotherlanguage[script=cyrillic]{serbian}
\else
  \if@ftn@latin
    \setmainlanguage[script=latin]{serbian}
  \else
    \setmainlanguage[script=cyrillic]{serbian}
  \fi
  \setotherlanguage{english}
\fi

% Times New Roman / Arial / Courier New are not freely distributable
% fonts, so metric-compatible (and Cyrillic-capable) substitutes from
% the "Liberation fonts" family are used instead -- these ship with
% almost every TeX Live installation.
\setmainfont{Liberation Serif}
\newfontfamily\ftnsans{Liberation Sans}
\setmonofont{Liberation Mono}[Scale=0.92]

% If a real Times New Roman / Arial / Courier New is installed, simply
% comment out the three lines above and uncomment these instead:
% \setmainfont{Times New Roman}
% \newfontfamily\ftnsans{Arial}
% \setmonofont{Courier New}

%% ------------------------------------------------------------------
%%  Prelom teksta
%% ------------------------------------------------------------------
\RequirePackage{setspace}
\if@ftn@onehalf
  \onehalfspacing
\else
  \singlespacing
\fi
\setlength{\parindent}{1.25cm}
\setlength{\parskip}{0pt}
\clubpenalty=10000
\widowpenalty=10000

% Unlike the LaTeX default, the reference Word-based examples indent
% every paragraph uniformly, including the first one right after a
% chapter/section heading -- so disable LaTeX's usual "no indent after
% a heading" behaviour.
\let\@afterindentfalse\@afterindenttrue
\@afterindenttrue

\RequirePackage{ragged2e}
\RequirePackage{microtype}

%% ------------------------------------------------------------------
%%  Grafika, boje, hiperlinkovi
%% ------------------------------------------------------------------
\RequirePackage{graphicx}
\graphicspath{{./logo/}{./slike/}{./}}
\RequirePackage{xcolor}
\definecolor{ftnplava}{HTML}{0000EE}

% All link categories render in black -- this thesis is meant to be
% printed, and blue link text (the hyperref/LaTeX default) only helps
% on screen.
\RequirePackage[
  colorlinks=true,
  linkcolor=black,
  citecolor=black,
  urlcolor=black,
  filecolor=black,
  bookmarksnumbered=true,
  pdfstartview=FitH
]{hyperref}

%% ------------------------------------------------------------------
%%  Naslovi poglavlja / odeljaka
%%
%%  Primećeno u primerima: naslov poglavlja (\chapter) je podebljan,
%%  bezserifni font, ceo velikim slovima, poravnat udesno ("1. UVOD"),
%%  a naslovi odeljaka/pododeljaka su podebljani, bezserifni, poravnati
%%  ulevo, bez verzala ("4.2 Dijagram slučajeva korišćenja").
%% ------------------------------------------------------------------
\RequirePackage{titlesec}

\titleformat{\chapter}
  {\normalfont\ftnsans\bfseries\Large\raggedleft}
  {\thechapter.}{0.6em}{\MakeUppercase}
\titlespacing{\chapter}{0pt}{0pt}{40pt}

\titleformat{\section}
  {\normalfont\ftnsans\bfseries\large}
  {\thesection}{0.7em}{}
\titlespacing{\section}{0pt}{18pt}{8pt}

\titleformat{\subsection}
  {\normalfont\ftnsans\bfseries}
  {\thesubsection}{0.7em}{}
\titlespacing{\subsection}{0pt}{14pt}{6pt}

\titleformat{\subsubsection}
  {\normalfont\ftnsans\bfseries\itshape\normalsize}
  {\thesubsubsection}{0.7em}{}
\titlespacing{\subsubsection}{0pt}{12pt}{4pt}

\setcounter{secnumdepth}{4}
\setcounter{tocdepth}{3}

%% ------------------------------------------------------------------
%%  Sadržaj (TOC)
%% ------------------------------------------------------------------
\RequirePackage{tocloft}
\renewcommand{\cftchapfont}{\normalfont}
\renewcommand{\cftchappagefont}{\normalfont}
\renewcommand{\cftchapleader}{\cftdotfill{\cftdotsep}}
\renewcommand{\cftchapaftersnum}{.}
\renewcommand{\cftchapindent}{0pt}
\setlength{\cftbeforechapskip}{6pt}
\renewcommand{\cftsecindent}{1.3em}
\renewcommand{\cftsubsecindent}{2.8em}
\renewcommand{\cftsubsubsecindent}{4.3em}
\setlength{\cfttabindent}{0pt}
\setlength{\cftfigindent}{0pt}

%% ------------------------------------------------------------------
%%  Slike, tabele, primeri -- numeracija i naslovi (captions)
%% ------------------------------------------------------------------
\RequirePackage{caption}
\captionsetup{
  labelsep=colon,
  justification=centering,
  singlelinecheck=true,
  font=normalsize,
  labelfont=normalfont,
  skip=6pt
}

\RequirePackage{chngcntr}
\counterwithin{figure}{section}
\counterwithin{table}{section}
\counterwithin{equation}{section}

%% ------------------------------------------------------------------
%%  Listinzi (kod)
%%
%%  Uputstvo katedre traži belu pozadinu listinga, sintaksno
%%  obojen tekst i font Courier New za nazive klasa/metoda/atributa.
%% ------------------------------------------------------------------
\RequirePackage{listings}
\AtBeginDocument{\counterwithin{lstlisting}{section}}

\definecolor{lstBg}{HTML}{FFFFFF}
\definecolor{lstFrame}{HTML}{BFBFBF}
\definecolor{lstKeyword}{HTML}{0000FF}
\definecolor{lstComment}{HTML}{2E8B57}
\definecolor{lstString}{HTML}{A31515}
\definecolor{lstNumber}{HTML}{098658}

\lstdefinelanguage{Rust}{
  morekeywords={as,break,const,continue,crate,dyn,else,enum,extern,%
    false,fn,for,if,impl,in,let,loop,match,mod,move,mut,pub,ref,%
    return,self,Self,static,struct,super,trait,true,type,unsafe,%
    use,where,while,async,await},
  sensitive=true,
  morecomment=[l]{//},
  morecomment=[s]{/*}{*/},
  morestring=[b]",
}

\lstdefinelanguage{TypeScript}{
  morekeywords={break,case,catch,class,const,continue,debugger,%
    default,delete,do,else,export,extends,false,finally,for,%
    function,if,implements,import,in,instanceof,interface,let,%
    new,null,package,private,protected,public,return,static,%
    super,switch,this,throw,true,try,typeof,var,void,while,%
    with,yield,readonly,namespace,declare,type,as,enum,async,await},
  sensitive=true,
  morecomment=[l]{//},
  morecomment=[s]{/*}{*/},
  morestring=[b]",
  morestring=[b]',
  morestring=[b]`,
}

\lstset{
  backgroundcolor=\color{lstBg},
  basicstyle=\ttfamily\footnotesize,
  keywordstyle=\color{lstKeyword}\bfseries,
  commentstyle=\color{lstComment}\itshape,
  stringstyle=\color{lstString},
  numberstyle=\tiny\color{lstNumber},
  numbers=left,
  numbersep=8pt,
  frame=single,
  rulecolor=\color{lstFrame},
  framerule=0.4pt,
  breaklines=true,
  breakatwhitespace=false,
  showstringspaces=false,
  tabsize=2,
  captionpos=b,
  xleftmargin=12pt,
  xrightmargin=6pt,
}
\if@ftn@english
  \renewcommand{\lstlistingname}{Listing}
\else
  \renewcommand{\lstlistingname}{Листинг}
\fi

% So that \ref to a listing reads as "Listing 5.2.1" in running text:
\AtBeginDocument{%
  \providecommand{\lstlistingautorefname}{\lstlistingname}
}

%% ------------------------------------------------------------------
%%  Inline code / foreign terms
%% ------------------------------------------------------------------
% For class/method/attribute names etc. in running text (Courier New font)
\newcommand{\kod}[1]{\texttt{#1}}
\let\code\kod
% For foreign-script terms embedded in the surrounding text (e.g.
% technology names left in Latin script inside Cyrillic text) --
% set in italics, as in the reference examples.
\newcommand{\strterm}[1]{\emph{#1}}

%% ------------------------------------------------------------------
%%  Lists
%% ------------------------------------------------------------------
\RequirePackage{enumitem}
\setlist[itemize]{label=--,leftmargin=1.5em}
% Note (see "README za autore"): in the department's Serbian-language
% convention, list items start with a lowercase letter and end with a
% comma, the second-to-last item ends with " and", and the last item
% ends with a period. This is a manual writing convention, not
% something LaTeX enforces automatically.

%% ------------------------------------------------------------------
%%  Zaglavlje / podnožje strane
%%
%%  U primerima broj strane stoji u gornjem levom uglu, bez linije
%%  zaglavlja i bez teksta poglavlja u zaglavlju.
%% ------------------------------------------------------------------
\RequirePackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\thepage}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}
\fancypagestyle{plain}{\fancyhf{}\fancyhead[L]{\thepage}\renewcommand{\headrulewidth}{0pt}}

%% ------------------------------------------------------------------
%%  Thesis metadata (filled in by the student in the preamble)
%% ------------------------------------------------------------------
\if@ftn@english
  \newcommand{\@ftn@univerzitet}{University of Novi Sad}
  \newcommand{\@ftn@fakultet}{Faculty of Technical Sciences, Novi Sad}
  \newcommand{\@ftn@autor}{Candidate's Name}
  \newcommand{\@ftn@naslov}{Thesis Title}
  \newcommand{\@ftn@vrstarada}{Bachelor Thesis}
  \newcommand{\@ftn@studije}{Bachelor Academic Studies}
  \newcommand{\@ftn@studprogram}{Software Engineering and Information Technologies}
  \newcommand{\@ftn@oblast}{Field the thesis belongs to}
  \newcommand{\@ftn@mentor}{Dr Name Surname, title}
  \newcommand{\@ftn@predsednik}{Dr Name Surname, title}
  \newcommand{\@ftn@clankomisije}{Dr Name Surname, title}
  \newcommand{\@ftn@jezikpub}{english}
\else
  \newcommand{\@ftn@univerzitet}{Универзитет у Новом Саду}
  \newcommand{\@ftn@fakultet}{Факултет техничких наука у Новом Саду}
  \newcommand{\@ftn@autor}{Име и презиме кандидата}
  \newcommand{\@ftn@naslov}{Наслов дипломског рада}
  \newcommand{\@ftn@vrstarada}{Дипломски рад}
  \newcommand{\@ftn@studije}{Основне академске студије}
  \newcommand{\@ftn@studprogram}{Софтверско инжењерство и информационе технологије}
  \newcommand{\@ftn@oblast}{Област из које је рад}
  \newcommand{\@ftn@mentor}{др Име Презиме, звање}
  \newcommand{\@ftn@predsednik}{др Име Презиме, звање}
  \newcommand{\@ftn@clankomisije}{др Име Презиме, звање}
  \newcommand{\@ftn@jezikpub}{српски}
\fi

\newcommand{\univerzitet}[1]{\renewcommand{\@ftn@univerzitet}{#1}}
\newcommand{\fakultet}[1]{\renewcommand{\@ftn@fakultet}{#1}}
\newcommand{\autor}[1]{\renewcommand{\@ftn@autor}{#1}}
% Latin-script rendering of the author's name, used only on the (always
% English) KWD page -- \autor itself follows the thesis's own language/
% script, so on a Cyrillic thesis it is not usable there directly.
\newcommand{\@ftn@autorEN}{}
\newcommand{\autorEN}[1]{\renewcommand{\@ftn@autorEN}{#1}}

\newcommand{\naslov}[1]{\renewcommand{\@ftn@naslov}{#1}}
\newcommand{\@ftn@naslovEN}{}
\newcommand{\naslovEN}[1]{\renewcommand{\@ftn@naslovEN}{#1}}

\newcommand{\vrstarada}[1]{\renewcommand{\@ftn@vrstarada}{#1}}
% Serbian/English wording for the "Врста рада"/"contents code" field on
% the KDI page (always Serbian) and KWD page (always English)
% respectively -- independent of \vrstarada, which follows the thesis's
% own language and is used only on the title page.
\newcommand{\@ftn@vrstaradaSR}{Дипломски рад}
\newcommand{\vrstaradaSR}[1]{\renewcommand{\@ftn@vrstaradaSR}{#1}}
\newcommand{\@ftn@vrstaradaEN}{bachelor thesis}
\newcommand{\vrstaradaEN}[1]{\renewcommand{\@ftn@vrstaradaEN}{#1}}
\newcommand{\studije}[1]{\renewcommand{\@ftn@studije}{#1}}

\newcommand{\@ftn@mesto}{Novi Sad}
\newcommand{\mesto}[1]{\renewcommand{\@ftn@mesto}{#1}}

\newcommand{\@ftn@godina}{\the\year}
\newcommand{\godina}[1]{\renewcommand{\@ftn@godina}{#1}}

\newcommand{\studijskiprogram}[1]{\renewcommand{\@ftn@studprogram}{#1}}

\newcommand{\@ftn@brojindeksa}{SW xx/20xx}
\newcommand{\brojindeksa}[1]{\renewcommand{\@ftn@brojindeksa}{#1}}

\newcommand{\oblast}[1]{\renewcommand{\@ftn@oblast}{#1}}

\newcommand{\mentor}[1]{\renewcommand{\@ftn@mentor}{#1}}
\newcommand{\@ftn@mentorEN}{}
\newcommand{\mentorEN}[1]{\renewcommand{\@ftn@mentorEN}{#1}}

% Language of publication, ЈП field on the (Serbian) KDI page.
\newcommand{\jezikpublikacije}[1]{\renewcommand{\@ftn@jezikpub}{#1}}

\newcommand{\@ftn@naucnaoblast}{Софтверско инжењерство и информационе технологије}
\newcommand{\naucnaoblast}[1]{\renewcommand{\@ftn@naucnaoblast}{#1}}
\newcommand{\@ftn@naucnaoblastEN}{Software Engineering and Information Technologies}
\newcommand{\naucnaoblastEN}[1]{\renewcommand{\@ftn@naucnaoblastEN}{#1}}

\newcommand{\@ftn@naucnadisciplina}{Софтверско инжењерство}
\newcommand{\naucnadisciplina}[1]{\renewcommand{\@ftn@naucnadisciplina}{#1}}
\newcommand{\@ftn@naucnadisciplinaEN}{Software Engineering}
\newcommand{\naucnadisciplinaEN}[1]{\renewcommand{\@ftn@naucnadisciplinaEN}{#1}}

\newcommand{\@ftn@kljucnereci}{keyword 1, keyword 2, keyword 3}
\newcommand{\kljucnereci}[1]{\renewcommand{\@ftn@kljucnereci}{#1}}
\newcommand{\@ftn@keywords}{keyword 1, keyword 2, keyword 3}
\newcommand{\keywords}[1]{\renewcommand{\@ftn@keywords}{#1}}

\newcommand{\@ftn@izvod}{Abstract -- one paragraph describing the essence of the thesis: the problem, motivation, outline of the solution and the result.}
\newcommand{\izvod}[1]{\renewcommand{\@ftn@izvod}{#1}}
\newcommand{\@ftn@abstract}{Abstract -- one paragraph describing the essence of the thesis.}
\newcommand{\abstractEN}[1]{\renewcommand{\@ftn@abstract}{#1}}

\newcommand{\@ftn@brpoglavlja}{X}
\newcommand{\@ftn@brstranica}{X}
\newcommand{\@ftn@brcitata}{X}
\newcommand{\@ftn@brtabela}{X}
\newcommand{\@ftn@brslika}{X}
\newcommand{\@ftn@brgrafikona}{X}
\newcommand{\@ftn@brpriloga}{X}
\newcommand{\fizickiopis}[7]{%
  \renewcommand{\@ftn@brpoglavlja}{#1}%
  \renewcommand{\@ftn@brstranica}{#2}%
  \renewcommand{\@ftn@brcitata}{#3}%
  \renewcommand{\@ftn@brtabela}{#4}%
  \renewcommand{\@ftn@brslika}{#5}%
  \renewcommand{\@ftn@brgrafikona}{#6}%
  \renewcommand{\@ftn@brpriloga}{#7}%
}

%% Additional KDI/KWD table fields -- library-assigned numbers, format
%% codes, and standard boilerplate that some theses need to override.
%% Each has a default matching the usual department wording. Fields that
%% carry prose (rather than a shared code or number) get a separate EN
%% variant for the KWD page, since the KDI page is always Serbian and the
%% KWD page is always English regardless of the thesis's own language.
\newcommand{\@ftn@rednibroj}{}
\newcommand{\rednibroj}[1]{\renewcommand{\@ftn@rednibroj}{#1}}

\newcommand{\@ftn@identifikacionibroj}{}
\newcommand{\identifikacionibroj}[1]{\renewcommand{\@ftn@identifikacionibroj}{#1}}

\newcommand{\@ftn@udk}{}
\newcommand{\udk}[1]{\renewcommand{\@ftn@udk}{#1}}

\newcommand{\@ftn@tipdokumentacije}{монографска публикација}
\newcommand{\tipdokumentacije}[1]{\renewcommand{\@ftn@tipdokumentacije}{#1}}
\newcommand{\@ftn@tipdokumentacijeEN}{monographic publication}
\newcommand{\tipdokumentacijeEN}[1]{\renewcommand{\@ftn@tipdokumentacijeEN}{#1}}

\newcommand{\@ftn@tipzapisa}{текстуални штампани документ}
\newcommand{\tipzapisa}[1]{\renewcommand{\@ftn@tipzapisa}{#1}}
\newcommand{\@ftn@tipzapisaEN}{textual printed document}
\newcommand{\tipzapisaEN}[1]{\renewcommand{\@ftn@tipzapisaEN}{#1}}

\newcommand{\@ftn@jezikizvoda}{српски / енглески}
\newcommand{\jezikizvoda}[1]{\renewcommand{\@ftn@jezikizvoda}{#1}}
\newcommand{\@ftn@jezikizvodaEN}{Serbian / English}
\newcommand{\jezikizvodaEN}[1]{\renewcommand{\@ftn@jezikizvodaEN}{#1}}

\newcommand{\@ftn@zemljapublikovanja}{Република Србија}
\newcommand{\zemljapublikovanja}[1]{\renewcommand{\@ftn@zemljapublikovanja}{#1}}
\newcommand{\@ftn@zemljapublikovanjaEN}{Republic of Serbia}
\newcommand{\zemljapublikovanjaEN}[1]{\renewcommand{\@ftn@zemljapublikovanjaEN}{#1}}

\newcommand{\@ftn@geografskopodrucje}{АП Војводина}
\newcommand{\geografskopodrucje}[1]{\renewcommand{\@ftn@geografskopodrucje}{#1}}
\newcommand{\@ftn@geografskopodrucjeEN}{AP Vojvodina}
\newcommand{\geografskopodrucjeEN}[1]{\renewcommand{\@ftn@geografskopodrucjeEN}{#1}}

\newcommand{\@ftn@izdavac}{ауторски репринт}
\newcommand{\izdavac}[1]{\renewcommand{\@ftn@izdavac}{#1}}
\newcommand{\@ftn@izdavacEN}{author's reprint}
\newcommand{\izdavacEN}[1]{\renewcommand{\@ftn@izdavacEN}{#1}}

\newcommand{\@ftn@mestoiadresa}{Нови Сад, Факултет техничких наука, Трг Доситеја Обрадовића 6}
\newcommand{\mestoiadresa}[1]{\renewcommand{\@ftn@mestoiadresa}{#1}}
\newcommand{\@ftn@mestoiadresaEN}{Novi Sad, Faculty of Technical Sciences, Trg Dositeja Obradovića 6}
\newcommand{\mestoiadresaEN}[1]{\renewcommand{\@ftn@mestoiadresaEN}{#1}}

\newcommand{\@ftn@cuvase}{Библиотека Факултета техничких наука, Трг Доситеја Обрадовића 6, Нови Сад}
\newcommand{\cuvase}[1]{\renewcommand{\@ftn@cuvase}{#1}}
\newcommand{\@ftn@cuvaseEN}{Library of the Faculty of Technical Sciences, Trg Dositeja Obradovića 6, Novi Sad}
\newcommand{\cuvaseEN}[1]{\renewcommand{\@ftn@cuvaseEN}{#1}}

\newcommand{\@ftn@vaznanapomena}{}
\newcommand{\vaznanapomena}[1]{\renewcommand{\@ftn@vaznanapomena}{#1}}
\newcommand{\@ftn@vaznanapomenaEN}{}
\newcommand{\vaznanapomenaEN}[1]{\renewcommand{\@ftn@vaznanapomenaEN}{#1}}

\newcommand{\@ftn@datumprihvatanja}{}
\newcommand{\datumprihvatanja}[1]{\renewcommand{\@ftn@datumprihvatanja}{#1}}
\newcommand{\@ftn@datumodbrane}{}
\newcommand{\datumodbrane}[1]{\renewcommand{\@ftn@datumodbrane}{#1}}

\newcommand{\predsednikkomisije}[1]{\renewcommand{\@ftn@predsednik}{#1}}
\newcommand{\@ftn@predsednikEN}{}
\newcommand{\predsednikkomisijeEN}[1]{\renewcommand{\@ftn@predsednikEN}{#1}}

\newcommand{\clankomisije}[1]{\renewcommand{\@ftn@clankomisije}{#1}}
\newcommand{\@ftn@clankomisijeEN}{}
\newcommand{\clankomisijeEN}[1]{\renewcommand{\@ftn@clankomisijeEN}{#1}}

\newcommand{\@ftn@logouniverzitet}{univerzitet-logo}
\newcommand{\@ftn@logofakultet}{ftn-logo}

%% ------------------------------------------------------------------
%%  Header with logos (used on the title page and on official forms --
%%  assignment sheet, declaration, KDI)
%% ------------------------------------------------------------------
\newcommand{\ftnheader}{%
  \begin{center}
    \begin{minipage}[c]{0.16\textwidth}
      \centering\includegraphics[height=1.7cm]{\@ftn@logouniverzitet}
    \end{minipage}%
    \begin{minipage}[c]{0.68\textwidth}
      \centering\ftnsans
      {\@ftn@univerzitet}\\[2pt]
      \textbf{\MakeUppercase{\@ftn@fakultet}}
    \end{minipage}%
    \begin{minipage}[c]{0.16\textwidth}
      \centering\includegraphics[height=1.7cm]{\@ftn@logofakultet}
    \end{minipage}
  \end{center}
  \vspace{-4pt}
  \rule{\textwidth}{0.8pt}
}

%% ------------------------------------------------------------------
%%  Title page
%% ------------------------------------------------------------------
\renewcommand{\maketitle}{%
  \begin{titlepage}
    \thispagestyle{empty}
    \ftnheader
    \vspace{2.2cm}
    \begin{center}
      {\normalsize \@ftn@autor}\\[2.6cm]
      {\ftnsans\bfseries\Large \@ftn@naslov}\\[1.4cm]
      {\normalsize \@ftn@vrstarada}\\
      {\normalsize - \@ftn@studije\ -}
      \vfill
      {\normalsize \@ftn@mesto, \@ftn@godina.}
    \end{center}
  \end{titlepage}
}

%% ------------------------------------------------------------------
%%  Key documentation information (Serbian KDI form) / Key Words Documentation
%% ------------------------------------------------------------------
\RequirePackage{array}
\RequirePackage{tabularx}
\RequirePackage{colortbl}
\RequirePackage{arydshln}

\newcolumntype{L}{>{\raggedright\arraybackslash}X}

\newcommand{\@kdi@row}[2]{#1 & #2 \\ \hline}

%% Bordered header used by the KDI/KWD pages: logo, university/faculty name
%% and address on one line each, then a shaded title bar -- distinct from
%% \ftnheader (used by the title page), which keeps its own layout.
%% Takes the institution line, the address line, and the title-bar text as
%% explicit arguments (rather than branching on \if@ftn@english) because
%% the KDI page is always Serbian and the KWD page is always English,
%% regardless of which language the thesis itself is written in.
\newcommand{\@ftn@kdiheader}[3]{%
  \renewcommand{\arraystretch}{1.3}
  \noindent\begin{tabularx}{\textwidth}{|>{\centering\arraybackslash}m{2.2cm}|>{\centering\arraybackslash}X|}
    \hline
    \includegraphics[width=1.6cm]{\@ftn@logofakultet} &
      \ftnsans #1\newline
      \normalfont #2 \\
    \hline
    \multicolumn{2}{|c|}{\cellcolor{gray!15}\ftnsans\bfseries #3} \\
    \hline
  \end{tabularx}%
}

\newcommand{\KDIpage}{%
  \clearpage
  \thispagestyle{empty}
  \begingroup
  \setlength{\arrayrulewidth}{0.9pt}
  \setlength{\dashlinedash}{1pt}
  \setlength{\dashlinegap}{1pt}
  \@ftn@kdiheader{Универзитет у Новом Саду\ \textbullet\ \textbf{ФАКУЛТЕТ ТЕХНИЧКИХ НАУКА}}%
    {21000 Нови Сад, Трг Доситеја Обрадовића 6}%
    {КЉУЧНА ДОКУМЕНТАЦИЈСКА ИНФОРМАЦИЈА}
  \vspace{6pt}
  \small
  \renewcommand{\arraystretch}{1.25}
  \noindent\begin{tabularx}{\textwidth}{|>{\raggedright\arraybackslash}p{4.3cm}:L|}
    \hline
    Редни број, \textbf{РБР}: & \@ftn@rednibroj \\ \hdashline
    Идентификациони број, \textbf{ИБР}: & \@ftn@identifikacionibroj \\ \hdashline
    Тип документације, \textbf{ТД}: & \@ftn@tipdokumentacije \\ \hdashline
    Тип записа, \textbf{ТЗ}: & \@ftn@tipzapisa \\ \hdashline
    Врста рада, \textbf{ВР}: & \@ftn@vrstaradaSR \\ \hdashline
    Аутор, \textbf{АУ}: & \@ftn@autor \\ \hdashline
    Ментор, \textbf{МН}: & \@ftn@mentor \\ \hdashline
    Наслов рада, \textbf{НР}: & \@ftn@naslov \\ \hdashline
    Језик публикације, \textbf{ЈП}: & \@ftn@jezikpub \\ \hdashline
    Језик извода, \textbf{ЈИ}: & \@ftn@jezikizvoda \\ \hdashline
    Земља публиковања, \textbf{ЗП}: & \@ftn@zemljapublikovanja \\ \hdashline
    Уже географско подручје, \textbf{УГП}: & \@ftn@geografskopodrucje \\ \hdashline
    Година, \textbf{ГО}: & \@ftn@godina. \\ \hdashline
    Издавач, \textbf{ИЗ}: & \@ftn@izdavac \\ \hdashline
    Место и адреса, \textbf{МА}: & \@ftn@mestoiadresa \\ \hdashline
    Физички опис рада, \textbf{ФО}:\newline
      {\scriptsize(поглавља/страна/цитата/табела/слика/графика/прилога)} &
      \@ftn@brpoglavlja\ / \@ftn@brstranica\ / \@ftn@brcitata\ / \@ftn@brtabela\ /
      \@ftn@brslika\ / \@ftn@brgrafikona\ / \@ftn@brpriloga \\ \hdashline
    Научна област, \textbf{НО}: & \@ftn@naucnaoblast \\ \hdashline
    Научна дисциплина, \textbf{НД}: & \@ftn@naucnadisciplina \\ \hdashline
    Предметна одредница / кључне речи, \textbf{ПО}: & \@ftn@kljucnereci \\ \hdashline
    \textbf{УДК}: & \@ftn@udk \\ \hdashline
    Чува се, \textbf{ЧУ}: & \@ftn@cuvase \\ \hdashline
    Важна напомена, \textbf{ВН}: & \@ftn@vaznanapomena \\ \hdashline
    Извод, \textbf{ИЗ}: & \@ftn@izvod \\ \hdashline
    Датум прихватања теме, \textbf{ДП}: & \@ftn@datumprihvatanja \\ \hdashline
    Датум одбране, \textbf{ДО}: & \@ftn@datumodbrane \\ \hdashline
    Чланови комисије, \textbf{КО}: & \\ \hdashline
    \quad председник: & \@ftn@predsednik \\ \hdashline
    \quad члан: & \@ftn@clankomisije \\ \hdashline
    \quad члан, ментор: & \@ftn@mentor \\
    \hline
  \end{tabularx}
  \endgroup
}

\newcommand{\KWDpage}{%
  \clearpage
  \thispagestyle{empty}
  \begingroup
  \setlength{\arrayrulewidth}{0.9pt}
  \setlength{\dashlinedash}{1pt}
  \setlength{\dashlinegap}{1pt}
  \@ftn@kdiheader{University of Novi Sad\ \textbullet\ \textbf{FACULTY OF TECHNICAL SCIENCES}}%
    {21000 Novi Sad, Trg Dositeja Obradovića 6}%
    {KEY WORDS DOCUMENTATION}
  \vspace{6pt}
  \small
  \renewcommand{\arraystretch}{1.25}
  \noindent\begin{tabularx}{\textwidth}{|>{\raggedright\arraybackslash}p{4.3cm}:L|}
    \hline
    Accession number, \textbf{AN}: & \@ftn@rednibroj \\ \hdashline
    Identification number, \textbf{IN}: & \@ftn@identifikacionibroj \\ \hdashline
    Document type, \textbf{DT}: & \@ftn@tipdokumentacijeEN \\ \hdashline
    Type of record, \textbf{TR}: & \@ftn@tipzapisaEN \\ \hdashline
    Contents code, \textbf{CC}: & \@ftn@vrstaradaEN \\ \hdashline
    Author, \textbf{AU}: & \@ftn@autorEN \\ \hdashline
    Mentor, \textbf{MN}: & \@ftn@mentorEN \\ \hdashline
    Title, \textbf{TI}: & \@ftn@naslovEN \\ \hdashline
    Language of text, \textbf{LT}: & \if@ftn@english English\else Serbian\fi \\ \hdashline
    Language of abstract, \textbf{LA}: & \@ftn@jezikizvodaEN \\ \hdashline
    Country of publication, \textbf{CP}: & \@ftn@zemljapublikovanjaEN \\ \hdashline
    Locality of publication, \textbf{LP}: & \@ftn@geografskopodrucjeEN \\ \hdashline
    Publication year, \textbf{PY}: & \@ftn@godina. \\ \hdashline
    Publisher, \textbf{PB}: & \@ftn@izdavacEN \\ \hdashline
    Publication place, \textbf{PP}: & \@ftn@mestoiadresaEN \\ \hdashline
    Physical description, \textbf{PD}:\newline
      {\scriptsize(chapters/pages/ref./tables/pictures/graphs/appendixes)} &
      \@ftn@brpoglavlja\ / \@ftn@brstranica\ / \@ftn@brcitata\ / \@ftn@brtabela\ /
      \@ftn@brslika\ / \@ftn@brgrafikona\ / \@ftn@brpriloga \\ \hdashline
    Scientific field, \textbf{SF}: & \@ftn@naucnaoblastEN \\ \hdashline
    Scientific discipline, \textbf{SD}: & \@ftn@naucnadisciplinaEN \\ \hdashline
    Subject/Keywords, \textbf{S/KW}: & \@ftn@keywords \\ \hdashline
    \textbf{UC}: & \@ftn@udk \\ \hdashline
    Holding data, \textbf{HD}: & \@ftn@cuvaseEN \\ \hdashline
    Note, \textbf{N}: & \@ftn@vaznanapomenaEN \\ \hdashline
    Abstract, \textbf{AB}: & \@ftn@abstract \\ \hdashline
    Accepted by sci. Board on, \textbf{ASB}: & \@ftn@datumprihvatanja \\ \hdashline
    Defended on, \textbf{DE}: & \@ftn@datumodbrane \\ \hdashline
    Defense board, \textbf{DB}: & \\ \hdashline
    \quad president: & \@ftn@predsednikEN \\ \hdashline
    \quad member: & \@ftn@clankomisijeEN \\ \hdashline
    \quad member, mentor: & \@ftn@mentorEN \\
    \hline
  \end{tabularx}
  \endgroup
}

%% ------------------------------------------------------------------
%%  Inserting signed official forms (thesis assignment sheet, conflict-
%%  of-interest declaration, etc.) as a scanned PDF instead of retyping
%%  them by hand.
%% ------------------------------------------------------------------
\RequirePackage{pdfpages}
\newcommand{\ubaciobrazac}[1]{%
  \includepdf[pages=-,pagecommand={\thispagestyle{empty}}]{#1}%
}

\endinput
