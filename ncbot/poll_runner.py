import threading
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

# Set to store processed chat IDs
last_processed_chats = []

def run_async_task(chatC):
    # Run the asynchronous function within a separate thread
    asyncio.run(deal_unread_chat(chatC))

def start():
    global last_processed_chats  # Declare global variable
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
                
                for chat in sorted(unread_chats, key=lambda x: x['id']):
                    chatC = NCChat(chat)
                    nc_agent.mark_chat_read(chatC.conversation_token, chatC.chat_id)
                unread_chats = sorted(unread_chats, key=lambda x: x['id'])

                if last_processed_chats!=unread_chats:
                    for chat in unread_chats:
                        chatC = NCChat(chat)
                        thread = threading.Thread(target=run_async_task(chatC))
                        thread.daemon = True
                        thread.start()
                        last_processed_chats=unread_chats

        except Exception as e:
            traceback.print_exc()
            logger.error(e)
        time.sleep(ncconfig.cf.poll_interval_s)


async def deal_unread_chat(chatC):
    try:
        await commander.dispatch(chatC)
        nc_agent.send_message(chatC.conversation_token, chatC.chat_id,
                                chatC.response, chatC.chat_message, chatC.user_id, False)
    except Exception as e:
        traceback.print_exc()
        logger.error(e)