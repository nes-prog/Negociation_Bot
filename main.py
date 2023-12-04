from tkinter import *
from model_bert import *
import argparse
from negociation_products_operations import * 
from bot_responses_negotiations import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="train", help = 'training or test')
    args = parser.parse_args()
    if args.mode == "train":
        train_run()       
    else: 
        # GUI
        root = Tk()
        root.title("Chatbot")
        
        BG_GRAY = "#ABB2B9"
        BG_COLOR = "#17202A"
        TEXT_COLOR = "#EAECEE"
        
        FONT = "Helvetica 14"
        FONT_BOLD = "Helvetica 13 bold"
        
        # Send function
        def send():
            send = "You -> " + e.get()
            txt.insert(END, "\n" + send)
        
            user_input = e.get().lower()
            a = get_response(user_input, params['s'])
            txt.insert(END, "\n" + "Bot ->" + str(a))
        
            if (a == "Goodbye, have a nice day " ) or (a == "you're welcome ! ") or (a == "okey, Welcome!" ):
                root.after(1500,lambda:root.destroy())   
        

        
            e.delete(0, END)
            # return str(a)

        lable1 = Label(root, bg=BG_COLOR, fg=TEXT_COLOR, text="Welcome", font=FONT_BOLD, pady=10, width=20, height=1).grid(
            row=0)
        
        txt = Text(root, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT, width=60)
        txt.grid(row=1, column=0, columnspan=2)
        
        scrollbar = Scrollbar(txt)
        scrollbar.place(relheight=1, relx=0.974)
        
        e = Entry(root, bg="#2C3E50", fg=TEXT_COLOR, font=FONT, width=55)
        e.grid(row=2, column=0)
        send = Button(root, text="Send", font=FONT_BOLD, bg=BG_GRAY,
                    command=send).grid(row=2, column=1)
            
        root.mainloop()
