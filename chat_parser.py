import re
import pandas as pd

def parse_chat(chat_text):
    pattern = r'(\d{1,2}/\d{1,2}/\d{2}), (\d{1,2}:\d{2}\s?[apAP][mM]) - (.*?): (.*)'

    data = []
    for line in chat_text.split('\n'):
        match = re.match(pattern, line)
        if match:
            date_str = match.group(1)
            time_str = match.group(2).replace('\u202f', ' ')
            sender = match.group(3)
            message = match.group(4)

            datetime_str = f"{date_str} {time_str}"
            try:
                dt = pd.to_datetime(datetime_str, format='%d/%m/%y %I:%M %p')
            except:
                dt = pd.to_datetime(datetime_str)

            data.append({
                'datetime': dt,
                'date': dt.date(),
                'time': dt.time(),
                'hour': dt.hour,
                'sender': sender,
                'message': message
            })
        else:
            if data:
                data[-1]['message'] += '\n' + line

    df = pd.DataFrame(data)
    return df
