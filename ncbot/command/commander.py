from ncbot.nc_helper import NCHelper
from ncbot.nc_chat import NCChat
from ncbot.plugins.openai.chat import chat3
import os
import importlib.util

nc_agent = NCHelper()

current_command = {}

plugin_path = 'ncbot/plugins'

user_command_cache = {}

def get_default_desc():
    desc = "You should type !Plugin:Function to talk with me.\n\nCurrent supported plugins are:\n"
    for key in current_command:
        desc += key+'\n'
    desc += "\nType !Plugin to see detail about plugin.\n"
    desc += "The last command will be remembered if capable, so you should not type the command first next time."
    return desc


def get_plugin_desc(plname):
    desc = 'Supported commands are:\n'
    plugin = current_command[plname]
    for key in plugin:
        desc += f'{key}: {plugin[key]["desc"]}\n'
    desc += f'type !{plname}:command input to use it.'
    return desc

def dispatch(chat: NCChat):
    chat.response = chat3(chat.user_id, chat.user_name, chat.chat_message)

def register(plname, funcname, desc, func, remember_command):
    if plname in current_command:
        current_command[plname][funcname] = {'desc':desc, 'func':func, 'remember':remember_command}
    else:
        current_command[plname] = {funcname: {'desc':desc, 'func':func, 'remember':remember_command}}


def load_plugin(path):
    for filename in os.listdir(path):
        tmppath = os.path.join(path, filename)
        if os.path.isfile(tmppath):
            if filename.endswith('.py') and not filename.startswith('__init'):
                spec = importlib.util.spec_from_file_location(filename[:-3], os.path.join(path, filename))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
        elif os.path.isdir(tmppath):
            load_plugin(tmppath)
