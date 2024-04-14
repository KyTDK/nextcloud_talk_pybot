from ncbot.plugins.utils.history import get_instance, MemoryHistoryUtil
import ncbot.command.base as base

history_util = get_instance()

plugin_name = 'setting'


@base.command(plname=plugin_name, funcname='clear_history',desc='Delete all chat history.', remember_command=False)
def clear_history(chatid, username, input):
    history_util.clear_memory(chatid)
    return 'History cleared!'