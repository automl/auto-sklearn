# -*- encoding: utf-8 -*-
# The MIT License (MIT)
#
# Copyright (c) 2014 zorro3, bw2 and Pablosan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import argparse
import os
import re
import sys
import types
from collections import OrderedDict

from six.moves import cStringIO as StringIO


__version__ = '0.9.3'

ACTION_TYPES_THAT_DONT_NEED_A_VALUE = {
    argparse._StoreTrueAction, argparse._StoreFalseAction,
    argparse._CountAction, argparse._StoreConstAction,
    argparse._AppendConstAction
}

# global ArgumentParser instances
_parsers = {}


def initArgumentParser(name=None, **kwargs):
    """Creates a global ArgumentParser instance with the given name, passing
    any args other than "name" to the ArgumentParser constructor.

    This instance can then be retrieved using getArgumentParser(..)

    """

    if name is None:
        name = 'default'

    if name in _parsers:
        raise ValueError(("kwargs besides 'name' can only be passed in the"
                          " first time. '%s' ArgumentParser already exists: %s"
                          ) % (name, _parsers[name]))

    kwargs.setdefault('formatter_class',
                      argparse.ArgumentDefaultsHelpFormatter)
    kwargs.setdefault('conflict_handler', 'resolve')
    _parsers[name] = ArgumentParser(**kwargs)


def getArgumentParser(name=None, **kwargs):
    """Returns the global ArgumentParser instance with the given name.

    The 1st time this function is called, a new ArgumentParser instance
    will be created for the given name, and any args other than "name"
    will be passed on to the ArgumentParser constructor.

    """
    if name is None:
        name = 'default'

    if len(kwargs) > 0 or name not in _parsers:
        initArgumentParser(name, **kwargs)

    return _parsers[name]


class ArgumentDefaultsRawHelpFormatter(argparse.ArgumentDefaultsHelpFormatter,
                                       argparse.RawTextHelpFormatter,
                                       argparse.RawDescriptionHelpFormatter):

    """HelpFormatter that adds default values AND doesn't do line-wrapping"""


class ArgumentParser(argparse.ArgumentParser):

    """Drop-in replacement for argparse.ArgumentParser that adds support for
    environment variables and .ini or .yaml-style config files.
    """

    def __init__(self,
                 prog=None,
                 usage=None,
                 description=None,
                 epilog=None,
                 version=None,
                 parents=[],
                 formatter_class=argparse.HelpFormatter,
                 prefix_chars='-',
                 fromfile_prefix_chars=None,
                 argument_default=None,
                 conflict_handler='error',
                 add_help=True,
                 add_config_file_help=True,
                 add_env_var_help=True,
                 default_config_files=[],
                 allow_unknown_config_file_keys=False,
                 args_for_setting_config_path=[],
                 config_arg_is_required=False,
                 config_arg_help_message='config file path', ):
        """Supports all the same args as the argparse.ArgumentParser
        constructor, as well as the following additional args.

        Additional Args:
            add_config_file_help: Whether to add a description of config file
                syntax to the help message.
            add_env_var_help: Whether to add something to the help message for
                args that can be set through environment variables.
            default_config_files: When specified, this list of config files will
                be parsed in order, with the values from each config file
                taking precedence over pervious ones. This allows an application
                to look for config files in multiple standard locations such as
                the install directory, home directory, and current directory:
                ["<install dir>/app_config.ini",
                "~/.my_app_config.ini",
                "./app_config.txt"]
            allow_unknown_config_file_keys: Whether unknown config file keys
                should be ignored or whether it should be an error.
            args_for_setting_config_path: A list of one or more command line
                args that would allow a user to provide a config file path
                (eg. ["-c", "--config-file"]). Default: []
            config_arg_is_required: when args_for_setting_config_path is set,
                set this to True to always require users to provide a config path.
            config_arg_help_message: when args_for_setting_config_path is set,
                this will be the help message for the config_file_args.

        """
        self._add_config_file_help = add_config_file_help
        self._add_env_var_help = add_env_var_help

        # extract kwargs that can be passed to the super constructor
        kwargs_for_super = {
            k: v
            for k, v in locals().items()
            if k in ['prog', 'usage', 'description', 'epilog', 'version', 'parents',
                     'formatter_class', 'prefix_chars', 'fromfile_prefix_chars',
                     'argument_default', 'conflict_handler', 'add_help']
        }
        if sys.version_info >= (3, 3) and 'version' in kwargs_for_super:
            del kwargs_for_super['version']  # version arg deprecated in v3.3

        argparse.ArgumentParser.__init__(self, **kwargs_for_super)

        # parse the additionial args
        self._default_config_files = default_config_files
        self._allow_unknown_config_file_keys = allow_unknown_config_file_keys
        if args_for_setting_config_path:
            self.add_argument(*args_for_setting_config_path,
                              dest='config_file',
                              required=config_arg_is_required,
                              help=config_arg_help_message,
                              is_config_file=True)

    def parse_args(self,
                   args=None,
                   namespace=None,
                   config_file_contents=None,
                   env_vars=os.environ):
        """Supports all the same args as the ArgumentParser.parse_args(..), as
        well as the following additional args.

        Additional Args:
            args: a list of args as in argparse, or a string (eg. "-x -y bla")
            config_file_contents: String. Used for testing.
            env_vars: Dictionary. Used for testing.

        """
        args, argv = self.parse_known_args(
            args=args,
            namespace=namespace,
            config_file_contents=config_file_contents,
            env_vars=env_vars)
        if argv:
            self.error('unrecognized arguments: %s' % ' '.join(argv))
        return args

    def parse_known_args(self,
                         args=None,
                         namespace=None,
                         config_file_contents=None,
                         env_vars=os.environ):
        """Supports all the same args as the ArgumentParser.parse_args(..), as
        well as the following additional args.

        Additional Args:
            args: a list of args as in argparse, or a string (eg. "-x -y bla")
            config_file_contents: String. Used for testing.
            env_vars: Dictionary. Used for testing.

        """

        if args is None:
            args = sys.argv[1:]
        elif isinstance(args, str):
            args = args.split()
        else:
            args = list(args)

        # maps string describing the source (eg. env var) to a settings dict
        # to keep track of where values came from (used by print_values())
        self._source_to_settings = OrderedDict()
        self._command_line_args_string = ' '.join(args)
        if args:
            self._source_to_settings['Command Line Args: '] = {
                '': self._command_line_args_string
            }

        # add env var settings to the command line that aren't there already
        env_var_args = []
        actions_with_env_var_values = [
            a for a in self._actions
            if a.option_strings and a.env_var and a.env_var in env_vars and not
            any(opt in args for opt in a.option_strings)
        ]
        for a in actions_with_env_var_values:
            key = a.env_var
            value = env_vars[key]
            env_var_args += self.convert_setting_to_command_line_arg(a, key,
                                                                     value)

        args = env_var_args + args

        if env_var_args:
            self._source_to_settings['Environment Variables:\n'] = OrderedDict(
                [(a.env_var, env_vars[a.env_var])
                 for a in actions_with_env_var_values])

        # read settings from config file(s)
        if config_file_contents:
            stream = StringIO(config_file_contents)
            stream.name = 'method arg'
            config_streams = [stream]
        else:
            config_streams = self._open_config_files(args)

        # add config file settings to the command line that aren't there
        # already

        # for each action, add its possible config keys to a dict
        possible_config_keys = {
            config_key: action
            for action in self._actions
            for config_key in self.get_possible_config_keys(action)
        }

        # parse each config file
        for stream in config_streams[::-1]:
            try:
                config_settings = self.parse_config_file(stream)
            finally:
                if hasattr(stream, 'close'):
                    stream.close()
            # make sure config file doesn't use any unknown keys
            if not self._allow_unknown_config_file_keys:
                invalid_keys = list(set(config_settings.keys()) -
                                    set(possible_config_keys.keys()))
                if invalid_keys:
                    self.error(('%s contains unknown config key(s): %s') %
                               (stream.name, ', '.join(invalid_keys)))

            # add config settings to the command line if they aren't there
            # already
            config_args = []
            for key, value in config_settings.items():
                if key in possible_config_keys:
                    action = possible_config_keys[key]
                    already_on_command_line = any(
                        arg in args for arg in action.option_strings)
                    if already_on_command_line:
                        del config_settings[key]
                    else:
                        config_args += self.convert_setting_to_command_line_arg(
                            action, key, value)

            args = config_args + args

            if config_args:
                self._source_to_settings[
                    'Config File (%s):\n' %
                    stream.name] = config_settings

        # save default settings for use by print_values()
        default_settings = OrderedDict()
        for a in self._actions:
            already_on_command_line = any(arg in args
                                          for arg in a.option_strings)
            cares_about_default = a.option_strings or a.nargs in [
                argparse.OPTIONAL, argparse.ZERO_OR_MORE
            ]
            if (already_on_command_line or not cares_about_default or
                    a.default is None or a.default == argparse.SUPPRESS or
                    type(a) in ACTION_TYPES_THAT_DONT_NEED_A_VALUE):
                continue
            else:
                key = a.option_strings[-1] if a.option_strings else a.dest
                default_settings[key] = str(a.default)

        if default_settings:
            self._source_to_settings['Defaults:\n'] = default_settings

        # parse all args (including command-line, config file, and env var)
        return argparse.ArgumentParser.parse_known_args(self,
                                                        args=args,
                                                        namespace=namespace)

    def parse_config_file(self, stream):
        """Parses a config file and return a dictionary of settings."""

        settings = OrderedDict()
        for i, line in enumerate(stream):
            line = line.strip()
            if not line or line[0] in [
                    '#', ';', '['] or line.startswith('---'):
                continue
            white_space = '\\s*'
            key = '(?P<key>[^:=;#\s]+?)'
            value1 = white_space + '[:=]' + white_space + '(?P<value>[^;#]+?)'
            value2 = white_space + '[\s]' + \
                white_space + '(?P<value>[^;#\s]+?)'
            comment = white_space + '(?P<comment>\\s[;#].*)?'

            key_only_match = re.match('^' + key + comment + '$', line)
            if key_only_match:
                key = key_only_match.group('key')
                settings[key] = 'true'
                continue

            key_value_match = re.match('^' + key + value1 + comment + '$',
                                       line) or \
                re.match('^' + key + value2 + comment + '$', line)
            if key_value_match:
                key = key_value_match.group('key')
                value = key_value_match.group('value')
                settings[key] = value
                continue

            self.error('Unexpected line %s in %s: %s' % (i, stream.name, line))
        return settings

    def convert_setting_to_command_line_arg(self, action, key, value):
        """Converts a config file or env var key/value to a list of command
        line args to append to the command line.

        Args:
            action: The action corresponding to this setting
            key: The config file key or env var name (used for error messages)
            value: The raw value string from the config file or env var

        """
        assert isinstance(value, str)

        args = []
        if value.lower() == 'true':
            if type(action) not in ACTION_TYPES_THAT_DONT_NEED_A_VALUE:
                self.error("%s set to 'True' rather than a value" % key)
            args.append(action.option_strings[-1])
        elif value.startswith('[') and value.endswith(']'):
            if not isinstance(action, argparse._AppendAction):
                self.error(("%s can't be set to a list '%s' unless its action "
                            "type is changed to 'append'") % (key, value))
            for list_elem in value[1:-1].split(','):
                args.append(action.option_strings[-1])
                args.append(list_elem.strip())
        else:
            if type(action) in ACTION_TYPES_THAT_DONT_NEED_A_VALUE:
                self.error("%s is a flag but is being set to '%s'" % (key,
                                                                      value))
            args.append(action.option_strings[-1])
            args.append(value)
        return args

    def get_possible_config_keys(self, action):
        """This method decides which actions can be set in a config file and
        what their keys will be.

        It return a list of 0 or more config keys that can be used to
        set the given action's value in a config file.

        """
        keys = []
        for arg in action.option_strings:
            if arg.startswith(2 * self.prefix_chars[0]):
                keys += [arg[2:],
                         arg]  # eg. for '--bla' return ['bla', '--bla']

        return keys

    def _open_config_files(self, command_line_args):
        """Tries to parse config file path(s) from within command_line_args.
        Returns a list of opened config files, including files specified on the
        command line as well as any default_config_files specified in the
        constructor that are present on disk.

        Args:
            command_line_args: List of all args (already split on spaces)

        """
        # open any default config files
        config_files = [
            open(f)
            for f in map(os.path.expanduser, self._default_config_files)
            if os.path.isfile(f)
        ]

        if not command_line_args:
            return config_files

        # list actions which had is_config_file=True set. Its possible there is
        # more than one such arg (perhaps to have multiple aliases for the
        # file)
        user_config_file_arg_actions = [a for a in self._actions
                                        if getattr(a, 'is_config_file', False)]

        if not user_config_file_arg_actions:
            return config_files

        for action in user_config_file_arg_actions:
            # try to parse out the config file path by using a clean new
            # ArgumentParser that only knows this one arg/action.
            arg_parser = argparse.ArgumentParser(
                prefix_chars=self.prefix_chars,
                add_help=False)

            arg_parser._add_action(action)

            # make parser not exit on error by replacing its error method.
            # Otherwise it sys.exits(..) if, for example, config file
            # is_required=True and user doesn't provide it.
            def error_method(self, message):
                pass

            arg_parser.error = types.MethodType(error_method, arg_parser)

            # check whether the user provided a value
            parsed_arg = arg_parser.parse_known_args(args=command_line_args)
            if not parsed_arg:
                continue
            namespace, _ = parsed_arg
            user_config_file = getattr(namespace, action.dest, None)
            if not user_config_file:
                continue
            # validate the user-provided config file path
            user_config_file = os.path.expanduser(user_config_file)
            if not os.path.isfile(user_config_file):
                self.error('File not found: %s' % user_config_file)

            config_files += [open(user_config_file)]

        return config_files

    def format_values(self):
        """Returns a string with all args and settings and where they came from
        (eg.

        command line, config file, enviroment variable or default)

        """

        r = StringIO()
        for source, settings in self._source_to_settings.items():
            r.write(source)
            for key, value in settings.items():
                if key:
                    r.write('  %-19s%s\n' % (key + ':', value))
                else:
                    r.write('  %s\n' % value)

        return r.getvalue()

    def print_values(self, file=sys.stdout):
        """Prints the format_values() string (to sys.stdout or another
        file)."""
        file.write(self.format_values())

    def format_help(self):
        msg = ''
        added_config_file_help = False
        added_env_var_help = False
        if self._add_config_file_help:
            default_config_files = self._default_config_files
            cc = 2 * self.prefix_chars[0]  # eg. --
            config_keys = [(arg, a)
                           for a in self._actions for arg in a.option_strings
                           if arg.startswith(cc) and a.dest != 'help']
            config_path_actions = [a for a in self._actions
                                   if getattr(a, 'is_config_file', False)]

            if (default_config_files or config_path_actions) and config_keys:
                self._add_config_file_help = False  # prevent duplication
                added_config_file_help = True

                msg += (
                    "Args that start with '%s' (eg. %s) can also be set in "
                    'a config file') % (cc, config_keys[0][0])
                config_arg_string = ' or '.join(a.option_strings[0]
                                                for a in config_path_actions
                                                if a.option_strings)
                if config_arg_string:
                    config_arg_string = 'specified via ' + config_arg_string
                if default_config_files or config_arg_string:
                    msg += ' (%s)' % ' or '.join(
                        default_config_files + [config_arg_string])
                msg += ' by using .ini or .yaml-style syntax '
                examples = []
                key_value_args = [
                    arg for arg, a in config_keys
                    if a.type not in ACTION_TYPES_THAT_DONT_NEED_A_VALUE
                ]
                if key_value_args:
                    examples += ['%s=value' % key_value_args[0].strip(cc)]
                flag_args = [arg for arg, a in config_keys
                             if a.type in ACTION_TYPES_THAT_DONT_NEED_A_VALUE]
                if flag_args:
                    examples += ['%s=TRUE' % flag_args[0].strip(cc)]
                if examples:
                    msg += '(eg. %s).' % ' or '.join(examples)

        if self._add_env_var_help:
            env_var_actions = [(a.env_var, a) for a in self._actions
                               if getattr(a, 'env_var', None)]
            for env_var, a in env_var_actions:
                env_var_help_string = '   [env var: %s]' % env_var
                if not a.help:
                    a.help = ''
                if env_var_help_string not in a.help:
                    a.help += env_var_help_string
                    added_env_var_help = True
                    self._add_env_var_help = False  # prevent duplication

        if added_env_var_help or added_config_file_help:
            value_sources = ['defaults']
            if added_config_file_help:
                value_sources = ['config file values'] + value_sources
            if added_env_var_help:
                value_sources = ['environment variables'] + value_sources
            msg += (' If an arg is specified in more than one place, then '
                    'command-line values override %s.') % (
                        ' which override '.join(value_sources)
            )
        if msg:
            self.description = (self.description or '') + ' ' + msg

        return argparse.ArgumentParser.format_help(self)


def add_argument(self, *args, **kwargs):
    """This method supports the same args as ArgumentParser.add_argument(..) as
    well as the additional args below.

    All
    Additional Args:
        env_var: The name of the environment variable to check.
        is_config_file: If True, this arg is treated as a config file path
            This provides an alternative way to specify config files in place of
            the ArgumentParser(fromfile_prefix_chars=..) mechanism.
            Default: False

    """

    env_var = kwargs.pop('env_var', None)
    is_config_file = kwargs.pop('is_config_file', None)

    action = self.original_add_argument_method(*args, **kwargs)

    is_positional_arg = not action.option_strings
    if is_positional_arg and env_var:
        raise ValueError("env_var can't be set for a positional arg.")
    if is_config_file and not isinstance(action, argparse._StoreAction):
        raise ValueError(
            "arg with is_config_file=True must have action='store'")

    action.env_var = env_var
    action.is_config_file = is_config_file

    return action

# wrap ArgumentParser's add_argument(..) method with the one above
argparse._ActionsContainer.original_add_argument_method = argparse._ActionsContainer.add_argument
argparse._ActionsContainer.add_argument = add_argument

# add all public classes in argparse module's namespace to this namespace so
# that the 2 modules are truly interchangeable
HelpFormatter = argparse.HelpFormatter
RawDescriptionHelpFormatter = argparse.RawDescriptionHelpFormatter
RawTextHelpFormatter = argparse.RawTextHelpFormatter
ArgumentDefaultsHelpFormatter = argparse.ArgumentDefaultsHelpFormatter
ArgumentError = argparse.ArgumentError
ArgumentTypeError = argparse.ArgumentTypeError
Action = argparse.Action
FileType = argparse.FileType
Namespace = argparse.Namespace

# create shorter aliases for the key methods and class names
getArgParser = getArgumentParser
getParser = getArgumentParser

ArgParser = ArgumentParser
Parser = ArgumentParser

argparse._ActionsContainer.add_arg = argparse._ActionsContainer.add_argument
argparse._ActionsContainer.add = argparse._ActionsContainer.add_argument

ArgumentParser.parse = ArgumentParser.parse_args
ArgumentParser.parse_known = ArgumentParser.parse_known_args

RawFormatter = RawDescriptionHelpFormatter
DefaultsFormatter = ArgumentDefaultsHelpFormatter
DefaultsRawFormatter = ArgumentDefaultsRawHelpFormatter
