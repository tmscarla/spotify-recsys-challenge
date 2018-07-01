import time
import logging
from telegram.bot import Bot
from functools import wraps
import threading

token = '512720388:AAHjYnJvvNld3rb70J1vp40gDEiRdcPHxsE'
chat_id = "-262107883"
# chat2 = "-314364535"
chat2 = "-1001356251815"

def bot_wrap(f):
    direct = Bot(token=token)

    def wrap(*args):

        direct.send_message(chat_id=chat_id, text="STARTED : "+f.__name__+" ")
        time1 = time.time()

        ret = f(*args)

        elapsed_time  = time.strftime("%H:%M:%S", time.gmtime(time.time()-time1))
        direct.send_message(chat_id=chat_id, text="ENDED : "+f.__name__+" in: "+str(elapsed_time)+
                                                            "\nturn off the server please <3")
        return ret
    return wrap


class Bot_v1(object):
    """
    only start and end of the program. just writes in telegram when the algorithm is lauchend and when is finished

    # EXAMPLE BOT V1
    bot = bot_v1("slim")
    bot.start()
    ====_ do things ====
    bot.end()
    """
    token = '512720388:AAHjYnJvvNld3rb70J1vp40gDEiRdcPHxsE'
    chat_id = "-262107883"

    def __init__(self, name, token="512720388:AAHjYnJvvNld3rb70J1vp40gDEiRdcPHxsE", chat_id="-262107883"):
        self.direct = Bot(token=token)
        self.chat_id = chat_id
        self.name = name
        self.start_time = time.time()

    def start(self ):
        self.direct.send_message(chat_id=self.chat_id, text="STARTED : "+self.name+" ")

    def end(self):
        elapsed_time  = time.strftime("%H:%M:%S", time.gmtime(time.time()-self.start_time))
        self.direct.send_message(chat_id=self.chat_id, text="ENDED : "+self.name+" in: "+str(elapsed_time)+
                                                            "\nturn off the server please <3")

    def send_message(self, text):
        self.direct.send_message(chat_id=self.chat_id, text=self.name+":\n"+text)


    def error(self, error_message):
        self.direct.send_message(chat_id=self.chat_id,text="ERROR : "+self.name+" blocked itself:\n"+error_message)

if __name__ == '__main__':

    # EXAMPLE BOT V1
    bot = Bot_v1("boh")
    bot.end()

# class bot_v2(object):
#
#     token = '512720388:AAHjYnJvvNld3rb70J1vp40gDEiRdcPHxsE'
#     chat_id = "-262107883"
#
#     def __init__(self, name):
#         self.direct = Bot(token=bot.token)
#         # self.updater = Updater(token=self.token)
#         self.name = name
#         self.logger = logging.getLogger(__name__)
#         self.updater = Updater(self.token)
#
#
#     def main(self):
#         """Start the bot."""
#         # Create the EventHandler and pass it your bot's token.
#
#         logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#                             level=logging.INFO)
#
#         dp = self.updater.dispatcher
#
#         dp.add_handler(CommandHandler("start", bot.start))
#         dp.add_handler(CommandHandler("help", bot.__help))
#         dp.add_handler(CommandHandler("status", bot.__status))
#
#         dp.add_error_handler(bot.__error)
#
#
#         print("0")
#         self.updater.start_polling()
#
#         print("1")
#         self.updater.idle()
#         print("2")
#
#
#     def end(self):
#         elapsed_time  = time.strftime("%H:%M:%S", time.gmtime(time.time()-self.start_time))
#         self.updater.bot.send_message(chat_id=bot.chat_id, text="ENDED :"+self.name+" "+str(elapsed_time))
#         self.updater.bot.send_message(chat_id=bot.chat_id, text="turn off the server please <3")
#
#     def update_status(self, message):
#         self.status_message = "LULLULASUDALSUDASLDUA LUL"
#
#
#     def __error(self, update, error):
#         self.logger.warning('Update "%s" caused error "%s"', update, error)
#
#     def __start(self, update,asdasdasd):
#         update.message.reply_text("STARTED :"+bot.name+" ")
#         bot.start_time = time.time()
#
#     def start(self, update):
#         """Send a message when the command /start is issued."""
#         update.message.reply_text('Hi!')
#
#     def __help(self, update):
#         """Send a message when the command /help is issued."""
#         update.message.reply_text(chat_id=self.chat_id, text='comandi: \help \status \n quando finisce te lo dice lui '
#                                                             'se hai messo la bot.end() ')
#
#     def __status(self,update):
#         update.message.reply_text("STATUS :"+self.name+"\n"+self.status_message)
#
# if __name__ == '__main__':
#
#     bot1 =  bot("slim?? sebaculo")
#     bot1.main()
#     print("1")
#     bot1.update_status("epoca sarcazzo")
#     bot1.end()





