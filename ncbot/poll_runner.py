import time
from ncbot.nc_helper import NCHelper
import traceback
from ncbot.nc_chat import NCChat
import ncbot.command.commander as commander
from ncbot.log_config import logger
import ncbot.config as ncconfig
import ncbot.nc_constants as ncconstants
import asyncio

nc_agent = NCHelper()


def start():

    while True:
        try:
            unread_chats = []
            unread_conversation = nc_agent.get_unread_conversation_list()
            logger.debug(
                f'found {len(unread_conversation)} unread conversations')
            for conversation in unread_conversation:
                if conversation['type'] == ncconstants.conversation_type_changelog:
                    continue
                chats = nc_agent.get_chat_list(
                    conversation['token'], conversation['unreadMessages'])
                unread_chats += chats
                logger.debug(
                    f'found {len(chats)} unread chats from token {conversation["token"]}')
            loop = asyncio.get_event_loop()
            loop.run_until_complete(deal_unread_chats(unread_chats))

        except Exception as e:
            traceback.print_exc()
            logger.error(e)
        time.sleep(ncconfig.cf.poll_interval_s)


async def deal_unread_chats(unread_chats):
    unread_chats = sorted(unread_chats, key=lambda x: x['id'])
    for chat in unread_chats:
        chatC = NCChat(chat)
        if chatC.user_id == ncconfig.cf.username:
            nc_agent.mark_chat_read(chatC.conversation_token, chatC.chat_id)
        else:
            try:
                nc_agent.mark_chat_read(
                    chatC.conversation_token, chatC.chat_id)
                await commander.dispatch(chatC)
                nc_agent.send_message(chatC.conversation_token, chatC.chat_id,
                                      chatC.response, chatC.chat_message, chatC.user_id, False)
            except Exception as e:
                traceback.print_exc()
                logger.error(e)
