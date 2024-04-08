from ncbot.nc_helper import NCHelper
from ncbot.nc_chat import NCChat
import os
import importlib.util

nc_agent = NCHelper()

current_command = {}

plugin_path = 'ncbot/plugins'

user_command_cache = {}


class Command:

    def __init__(self, chat: NCChat):
        commandstr:str = chat.chat_message
        self.matched_func = True
        self.matched_plugin = True
        self.plname = "openai"
        self.funcname = "chat3"
        self.value = None
        self.user_id = chat.user_id
        self.user_name = chat.user_name
        if not commandstr.startswith('!'):
            return
        try:
            commandpair = commandstr.split(' ',1)
            commanddetail = commandpair[0][1:].split(':')
            self.plname = commanddetail[0]
            self.funcname = commanddetail[1]
            self.value = commandpair[1]
        except Exception:
            pass


        if self.plname in current_command:
                self.matched_plugin = True
                if self.funcname in current_command[self.plname]:
                    self.matched_func = True
                    self.func = current_command[self.plname][self.funcname]['func']       


    def execute(self):
        try:
            return self.func(self.user_id, self.user_name, self.value)
            print(self.value)
        except Exception as e:
            return 'Something wrong happened! Please try again later.'


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


def find_last_command(chat: NCChat):
    if not chat.chat_message.startswith('!'):
        key = f'command_{chat.user_id}'
        if key in user_command_cache:
            command = user_command_cache[key]
            chat.chat_message = f'{command} {chat.chat_message}'


def save_last_command(chat: NCChat, command: Command):
    if current_command[command.plname][command.funcname]['remember']:
        key = f'command_{chat.user_id}'
        user_command_cache[key] = f'!{command.plname}:{command.funcname}'
        return True
    return False


def dispatch(chat: NCChat):
    ret = 'test'
    #nc_agent.lock_conversation(chat.conversation_token)

    find_last_command(chat)
    command = Command(chat)
    if command.matched_func:
        ret = command.execute()
        save_last_command(chat, command)
    elif command.matched_plugin:
        ret = get_plugin_desc(command.plname)
    else:
        ret = get_default_desc()
    #nc_agent.unlock_conversation(chat.conversation_token)
    chat.response = ret


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
