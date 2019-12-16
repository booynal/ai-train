'''
该代码参考：《Python自然语言处理实战：核心技术与算法》第4章 词性标注与命名实体识别

创建时间：2019/12/15
'''


import re
from datetime import datetime, timedelta
from dateutil.parser import parse
import jieba.posseg as psg


CN_NUM = {
    '零': 0,
    '一': 1,
    '二': 2,
    '三': 3,
    '四': 4,
    '五': 5,
    '六': 6,
    '七': 7,
    '八': 8,
    '九': 9
}
CN_UNIT = {
    '十': 10,
    '百': 100,
    '千': 1000
}


def time_extract(text):
    time_result = []
    word = ''
    keyDate = {'今天':0, '明天':1, '后天':2}
    for k, v in psg.cut(text):
        if k in keyDate:
            if word != '':
                time_result.append(word)
                word = (datetime.today() + timedelta(days=keyDate.get(k, 0))).strftime('%Y-%m-%d')
            else:
                time_result.append(datetime.today().strftime('%Y-%m-%d'))
        elif word != '':
            # m数词(numeral的第3个字符)；t时间词(time的第1个字符)
            # fix: q为量词(quantity)，"号"与"吗"拼接在一起就会是q
            if v in ['m', 't', 'q']:
                word = word + k
            else:
                time_result.append(word)
                word = ''
        elif v in ['m', 't']:
            word = k
    if word != '':
        time_result.append(word)
    result = list(filter(lambda x:x is not None, [parse_datetime(check_time_valid(w)) for w in time_result]))
    return result


def check_time_valid(word):
    if re.match(r'\d+$', word) and len(word) <= 6:
        return None
    subWord = re.sub(r'[号|日]\d+$', '日', word)
    if subWord != word:
        return check_time_valid(subWord)
    else:
        return subWord

def parse_datetime(word):
    if word is None or word is '':
        return None

    try:
        dt = parse(word, fuzzy=True)
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        pattern = r'([0-9零一二三四五六七八九十]+年)?' \
                  r'([0-9零一二三四五六七八九十]+月)?' \
                  r'([0-9零一二三四五六七八九十]+[号日])?' \
                  r'([上中下午晚早]+)?' \
                  r'([0-9零一二三四五六七八九十]+[点:\.时])?' \
                  r'([0-9零一二三四五六七八九十]+[分:]?)?' \
                  r'([0-9零一二三四五六七八九十]+秒?)?'
        m = re.match(pattern, word)
        if m.group(0) is not None:
            time_map = {
                'year': m.group(1),
                'month': m.group(2),
                'day': m.group(3),
                'hour': m.group(5) if m.group(5) is not None else '00',
                'minute': m.group(6) if m.group(6) is not None else '00',
                'second': m.group(7) if m.group(7) is not None else '00'
            }
            params = {}
            for name in time_map:
                value = time_map[name]
                if value is not None and len(value) != 0:
                    tmp = None
                    if name == 'year':
                        tmp = year2number(value[:-1])
                    else:
                        tmp = cn2number(value[:-1])
                    if tmp is not None:
                        params[name] = int(tmp)
            target_time = datetime.today().replace(**params)
            is_pm = m.group(4)
            if is_pm is not None:
                if is_pm == '晚上' or is_pm == '下午' or is_pm == '中午':
                    hour = target_time.time().hour
                    if hour < 12:
                        target_time = target_time.replace(hour=hour + 12)
            return target_time.strftime('%Y-%m-%d %H:%M:%S')
        else:
            return None

def year2number(year):
    if year is None or year is '':
        return None

    tmp = ''
    for char in year:
        if char in CN_NUM.keys():
            tmp += str(CN_NUM[char])
        else:
            tmp += char
    m = re.match(r'\d+', tmp)
    if m:
        one = m.group(0)
        if len(one) == 2:
            return int(datetime.today().year / 100) * 100 + int(one)
        else:
            return int(one)
    else:
        return None

def cn2number(word):
    if word is None or word is '':
        return None

    m = re.match(r'\d+', word)
    if m:
        return int(m.group(0))

    result = 0
    unit = 1
    for one in word[::-1]:
        if one in CN_UNIT.keys():
            unit = CN_UNIT[one]
        elif one in CN_NUM.keys():
            num = CN_NUM[one]
            result += num * unit
        else:
            return None
    if result < unit:
        result += unit
    return result


if __name__ == '__main__':
    # ['我/r', '要/v', '今天/t', '住/v', '到/v', '明天/t', '下午/t', '3/m', '点/m']
    text = '我要住到明天下午3点'
    print(['{}/{}'.format(x, y) for x, y in (psg.cut(text))])
    print(text, time_extract(text), sep=': ')

    text = '预定28号的房间'
    print(['{}/{}'.format(x, y) for x, y in (psg.cut(text))])
    print(text, time_extract(text), sep=': ')

    text = '我要从26号下午4点住到11月2号'
    print(['{}/{}'.format(x, y) for x, y in (psg.cut(text))])
    print(text, time_extract(text), sep=': ')

    text = '我要预定今天到30号的房间'
    print(['{}/{}'.format(x, y) for x, y in (psg.cut(text))])
    print(text, time_extract(text), sep=': ')

    text = '今天30号吗？'
    print(['{}/{}'.format(x, y) for x, y in (psg.cut(text))])
    print(text, time_extract(text), sep=': ')

