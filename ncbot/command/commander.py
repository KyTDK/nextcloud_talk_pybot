import time
from ncbot.nc_helper import NCHelper
import traceback
from ncbot.nc_chat import NCChat
import ncbot.command.commander as commander
from ncbot.log_config import logger
import ncbot.config as ncconfig
import ncbot.nc_constants as ncconstants

nc_agent = NCHelper()

def start():
    while True:
        try:
            unread_chats = []
            unread_conversations = nc_agent.get_unread_conversation_list()
            logger.debug(f'found {len(unread_conversations)} unread conversations')
            for conversation in unread_conversations:
                if conversation['type'] == ncconstants.conversation_type_changelog:
                    continue
                chats = nc_agent.get_chat_list(conversation['token'], conversation['unreadMessages'])
                unread_chats += chats
                logger.debug(f'found {len(chats)} unread chats from token {conversation["token"]}')
            deal_unread_chats(unread_chats)

        except Exception as e:
            traceback.print_exc()
            logger.error(e)
        time.sleep(ncconfig.cf.poll_interval_s)


def deal_unread_chats(unread_chats):
    unread_chats = sorted(unread_chats, key=lambda x: x['id'])
    for chat in unread_chats:
        chatC = NCChat(chat)
        if chatC.user_id == ncconfig.cf.username:
            skip_self_unread(chatC)
        else:
            try:
                commander.dispatch(chatC)
                send_response_appropriately(chatC)  # Use the new function
            except Exception as e:
                traceback.print_exc()
                logger.error(e)


def skip_self_unread(chat: NCChat):
    nc_agent.mark_chat_read(chat.conversation_token, chat.chat_id)


def send_response_appropriately(chat: NCChat):
    if chat.chat_type == ncconstants.chat_type_user:  # Direct message
        nc_agent.send_message(chat.conversation_token, chat.chat_id, chat.response, chat.chat_message, chat.user_id)
    else:  # Non-direct message, send to the channel
        nc_agent.send_message(chat.conversation_token,  # Use conversation_token for channel
                              chat.chat_id,  # Assuming chat_id identifies the channel message
                              chat.response,
                              chat.chat_message)  # user_id not needed for channel message

    nc_agent.mark_chat_read(chat.conversation_token, chat.chat_id)
