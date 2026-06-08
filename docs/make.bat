@ECHO OFF
REM Minimal Sphinx build for Windows. Run `pip install -e .[docs,notebook]` first.
pushd %~dp0

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=.
set BUILDDIR=_build

if "%1" == "" goto html

%SPHINXBUILD% -b %1 %SOURCEDIR% %BUILDDIR%\%1 -W --keep-going
goto end

:html
%SPHINXBUILD% -b html %SOURCEDIR% %BUILDDIR%\html -W --keep-going

:end
popd
