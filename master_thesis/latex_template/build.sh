#!/bin/bash
# ====================================================================
#    The author of this file licenses it to you under the Apache
#    License, Version 2.0. You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing,
#    software distributed under the License is distributed on an
#    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#    KIND, either express or implied.  See the License for the
#    specific language governing permissions and limitations
#    under the License.
# ====================================================================
####################
# Latex Build Script by Dietmar Malli (2017-11-07)
####################
FILENAME_WITHOUT_EXTENSION=Main
FILENAME="$FILENAME_WITHOUT_EXTENSION.tex"

#Test for needed tools:
PROG_AVAIL=$(which pdflatex | wc -l)
if [ "$PROG_AVAIL" -eq "0" ]; then
  echo Warning: pdflatex not found, but needed by this build system.
  exit -1
fi
PROG_AVAIL=$(which biber | wc -l)
if [ "$PROG_AVAIL" -eq "0" ]; then
  echo Warning: biber not found, but needed by this build system.
  exit -1
fi
PROG_AVAIL=$(which makeglossaries | wc -l)
if [ "$PROG_AVAIL" -eq "0" ]; then
  echo Warning: makeglossaries not found, but needed by this build system.
  exit -1
fi
PROG_AVAIL=$(which find | wc -l)
if [ "$PROG_AVAIL" -eq "0" ]; then
  echo Warning: find not found, but needed by this build system.
  exit -1
fi
PROG_AVAIL=$(which perl | wc -l)
if [ "$PROG_AVAIL" -eq "0" ]; then
  echo Warning: perl not found, could be needed by biber, but is sometimes statically linked / inbuilt into biber executeable...
fi

echo '---------------------------------'
echo 'Running pdflatex (1)...'
pdflatex -synctex=1 -interaction=nonstopmode $FILENAME

echo '---------------------------------'
echo 'Running biber: runs over all .bcf (Biber Control Files)...'
find . -name "*.bcf" -exec biber {} \;

echo '---------------------------------'
echo 'Running makeglossaries...'
makeglossaries $FILENAME_WITHOUT_EXTENSION #needs perl

echo '---------------------------------'
echo 'Running pdflatex (2)...'
pdflatex -synctex=1 -interaction=nonstopmode $FILENAME

echo '---------------------------------'
echo 'Running pdflatex...'
pdflatex -synctex=1 -interaction=nonstopmode $FILENAME

#recursive delete of temporary files:
find . -name "*.aux" -delete
find . -name "*eps-converted-to.pdf" -delete
find . -name "*.lof" -delete
find . -name "*.out" -delete
find . -name "*.toc" -delete
find . -name "*.run.xml" -delete
find . -name "*.lot" -delete
find . -name "*.gz" -delete
find . -name "*-blx.bib" -delete #biblatex
find . -name "*.aux.blg" -delete #biblatex
find . -name "*.bbl" -delete #biber
find . -name "*.bcf" -delete #biber
find . -name "*.blg" -delete #biber
find . -name "*.acn" -delete #makeglossaries
find . -name "*.acr" -delete #makeglossaries
find . -name "*.alg" -delete #makeglossaries
find . -name "*.ist" -delete #makeglossaries
find . -name "*.idx" -delete #makeindex
find . -name "*.dvi" -delete
find . -name "*.log" ! -name "$FILENAME_WITHOUT_EXTENSION.log" -delete
find . -name "*.fdb_latexmk" -delete #?Latex IDE?
find . -name "*.fls" -delete #?Latex IDE?

#Buildcount update:
#echo `date` > .lastCompiled
#read build < .buildNumber
#build=$((build+1))
#echo $build > .buildNumber
