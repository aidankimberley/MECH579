# How to Export Python Code to PDF

## Quick Summary
You created a LaTeX file (`.tex`) that formats and includes your Python code with syntax highlighting, then compiled it to PDF using `pdflatex`.

---

## Method 1: Using LaTeX (Professional Results) ⭐

### Prerequisites
- **LaTeX Distribution**: You already have MiKTeX installed (Windows). Other options:
  - Mac: MacTeX
  - Linux: TeX Live
  - Online: Overleaf (no installation needed)

### Step-by-Step Guide

#### 1. Create a LaTeX File (e.g., `my_code.tex`)

```latex
\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[margin=1in]{geometry}
\usepackage{listings}
\usepackage{xcolor}

% Define colors for syntax highlighting
\definecolor{codegreen}{rgb}{0,0.6,0}
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{codepurple}{rgb}{0.58,0,0.82}
\definecolor{backcolour}{rgb}{0.95,0.95,0.92}

% Configure Python syntax highlighting
\lstdefinestyle{pythonstyle}{
    backgroundcolor=\color{backcolour},   
    commentstyle=\color{codegreen},
    keywordstyle=\color{magenta},
    numberstyle=\tiny\color{codegray},
    stringstyle=\color{codepurple},
    basicstyle=\ttfamily\footnotesize,
    breakatwhitespace=false,         
    breaklines=true,                 
    keepspaces=true,                 
    numbers=left,                    
    numbersep=5pt,                  
    showspaces=false,                
    showstringspaces=false,
    showtabs=false,                  
    tabsize=4,
    language=Python,
    literate={Δ}{$\Delta$}1,
    extendedchars=true
}

\lstset{style=pythonstyle}

\title{My Python Code}
\author{Your Name}
\date{\today}

\begin{document}

\maketitle
\tableofcontents
\newpage

\section{First File}
\lstinputlisting[language=Python]{your_file.py}

\newpage
\section{Second File}
\lstinputlisting[language=Python]{another_file.py}

\end{document}
```

#### 2. Compile to PDF

Open PowerShell or Command Prompt in your project directory and run:

```powershell
pdflatex my_code.tex
```

**Important:** Run it **TWICE** if you have a table of contents:
```powershell
pdflatex my_code.tex
pdflatex my_code.tex
```

The first run generates the TOC, the second run includes it properly.

#### 3. Clean Up (Optional)

LaTeX creates auxiliary files. Delete them to keep things tidy:
```powershell
Remove-Item my_code.aux, my_code.log, my_code.toc
```

---

## Method 2: Using Python (Quick & Easy)

### Option A: Using `code2pdf` Package

```bash
pip install code2pdf
```

```bash
code2pdf your_file.py -o output.pdf
```

For multiple files:
```bash
code2pdf file1.py file2.py file3.py -o combined.pdf
```

### Option B: Using `pygments` and `reportlab`

```python
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
import pdfkit  # requires wkhtmltopdf

with open('your_code.py', 'r') as f:
    code = f.read()

highlighted_code = highlight(code, PythonLexer(), HtmlFormatter(full=True))
pdfkit.from_string(highlighted_code, 'output.pdf')
```

---

## Method 3: Using Online Tools (No Installation)

1. **Overleaf** (https://overleaf.com)
   - Create new project
   - Upload your `.py` files
   - Copy the LaTeX template above
   - Click "Recompile"
   - Download PDF

2. **Carbon** (https://carbon.now.sh)
   - Paste code
   - Style it
   - Export as PNG
   - Combine PNGs into PDF using online tools

---

## Customization Tips

### Change Font Size
```latex
basicstyle=\ttfamily\tiny,      % Very small
basicstyle=\ttfamily\footnotesize,  % Small (default in our template)
basicstyle=\ttfamily\small,     % Medium
basicstyle=\ttfamily\normalsize,    % Normal
```

### Remove Line Numbers
```latex
numbers=none,
```

### Change Color Scheme
```latex
\definecolor{backcolour}{rgb}{0.1,0.1,0.1}  % Dark background
basicstyle=\ttfamily\footnotesize\color{white},  % White text
```

### Add Line to Specific Line Range
```latex
\lstinputlisting[language=Python, firstline=10, lastline=50]{your_file.py}
```

### Different Languages
Change `language=Python` to:
- `language=Java`
- `language=C`
- `language=JavaScript`
- etc.

---

## Troubleshooting

### Error: "I can't find file..."
- Make sure you're in the correct directory
- Use `cd` to navigate to your project folder

### Unicode Errors (Δ, ∆, etc.)
Add these to your preamble:
```latex
\usepackage[T1]{fontenc}
literate={Δ}{$\Delta$}1,
```

### PDF Not Updating
- Make sure to close the PDF before recompiling
- Or use `-interaction=nonstopmode` flag:
  ```bash
  pdflatex -interaction=nonstopmode my_code.tex
  ```

### Long Lines Cut Off
Already handled with:
```latex
breaklines=true,
```

---

## Your Specific Setup

The file `brequet_range_optimizer.tex` in your project directory contains the template we used. You can:

1. **Modify it** to add/remove files
2. **Rename it** for different projects
3. **Copy it** as a template for future assignments

### To add more files:
Just add more sections before `\end{document}`:

```latex
\newpage
\section{New File}
\lstinputlisting[language=Python]{new_file.py}
```

### To compile:
```powershell
cd "C:\Users\aidan\Documents\McGill\MECH597\Assignment 2"
pdflatex brequet_range_optimizer.tex
pdflatex brequet_range_optimizer.tex
```

---

## Summary

**Easiest:** Use `code2pdf` Python package  
**Best Looking:** Use LaTeX (what we just did)  
**No Installation:** Use Overleaf online

For academic assignments, **LaTeX gives the most professional results** and that's what we used for your MECH597 assignment!

