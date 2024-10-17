#!/bin/sh

# This file is part of CorPipe <https://github.com/ufal/crac2024-corpipe>.
#
# Copyright 2024 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University in Prague, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

set -e

# Train and dev
wget https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-5478/CorefUD-1.2-public.zip
unzip CorefUD-1.2-public.zip
for f in CorefUD-1.2-public/data/*/*.conllu; do
  lang=$(basename $f)
  lang=${lang%%-*}
  mkdir -p $lang
  mv $f $lang/$(basename $f)
done
rm -r CorefUD-1.2-public/ CorefUD-1.2-public.zip

# Test data with zeros removed
mkdir test
(cd test
 wget http://ufal.mff.cuni.cz/~mnovak/files/corefud-1.2/test-blind.zip
 unzip test-blind.zip
 for f in *.conllu; do
   lang=${f%%-*}
   mv $f ../$lang/$f
 done
)
rm -rf test/

echo All done
