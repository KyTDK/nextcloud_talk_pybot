import threading
import time
from ncbot.nc_helper import NCHelper
import traceback
from ncbot.nc_chat import NCChat
import ncbot.command.commander as commander
from ncbot.log_config import logger
import ncbot.config as ncconfig
import ncbot.nc_constants as ncconstants

nc_agent = NCHelper()

def process_unread_chat(chat):
    chatC = NCChat(chat)
    try:
        nc_agent.mark_chat_read(chatC.conversation_token, chatC.chat_id)
        # Check if it's our message, otherwise process and reply
        if chatC.user_id != ncconfig.cf.username:
            commander.dispatch(chatC)
            nc_agent.send_message(chatC.conversation_token, chatC.chat_id, chatC.response, chatC.chat_message, chatC.user_id, False)
    except Exception as e:
        traceback.print_exc()
        logger.error(e)

def start():
    while True:
        try:
            unread_conversations = nc_agent.get_unread_conversation_list()
            logger.debug(f'found {len(unread_conversations)} unread conversations')
            threads = []
            for conversation in unread_conversations:
                if conversation['type'] == ncconstants.conversation_type_changelog:
                    continue
                chats = nc_agent.get_chat_list(conversation['token'],conversation['unreadMessages'])
                for chat in chats:
                    # Create a thread for each unread chat
                    thread = threading.Thread(target=process_unread_chat, args=(chat,))
                    threads.append(thread)
                    thread.start()

            # Wait for all threads to finish before fetching new conversations
            for thread in threads:
                thread.join()

        except Exception as e:
            traceback.print_exc()
            logger.error(e)

        time.sleep(ncconfig.cf.poll_interval_s)

if __name__ == "__main__":
    start()