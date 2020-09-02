# -*- encoding: utf-8 -*-
import re


SUBPATTERN = r'((?P<operation%d>==|>=|>|<)(?P<version%d>(\d+)?(\.[a-zA-Z0-9]+)?(\.\d+)?))'
RE_PATTERN = re.compile(
    r'^(?P<name>[\w\-]+)%s?(,%s)?$' % (SUBPATTERN % (1, 1), SUBPATTERN % (2, 2)))
