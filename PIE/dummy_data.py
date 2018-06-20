import random
import exrex

def gen_phone():
    format1 = '{} -X- -X- B-TEL\n- -X- -X- I-TEL\n{} -X- -X- I-TEL\n- -X- -X- I-TEL\n{} -X- -X- E-TEL\n' # 123-456-7890
    format4 = '{} -X- -X- B-TEL\n{} -X- -X- I-TEL\n{} -X- -X- E-TEL\n' # 123 456 7890
    format2 = '( -X- -X- B-TEL\n{} -X- -X- I-TEL\n) -X- -X- I-TEL\n{} -X- -X- I-TEL\n- -X- -X- I-TEL\n{} -X- -X- E-TEL\n' # (123) 456-7890
    format3 = '( -X- -X- B-TEL\n{}){} -X- -X- I-TEL\n- -X- -X- I-TEL\n{} -X- -X- E-TEL\n' # (123)456-7890
    format5 = '{} -X- -X- B-TEL\n. -X- -X- I-TEL\n{} -X- -X- I-TEL\n. -X- -X- I-TEL\n{} -X- -X- E-TEL\n' # 123.456.7890

    formatDict = {1:format1, 2:format2, 3:format3, 4:format4, 5:format5}

    first = str(random.randint(100,999))
    second = str(random.randint(1,999)).zfill(3)
    last = str(random.randint(1,9999)).zfill(4)

    return formatDict[random.randint(1, 4)].format(first, second, last)

def gen_sin():
    format_string = '{} -X- -X- B-SIN\n{} -X- -X- I-SIN\n{} -X- -X- E-SIN\n' # 123 456 789

    first = str(random.randint(1,999)).zfill(3)
    second = str(random.randint(1,999)).zfill(3)
    last = str(random.randint(1,999)).zfill(3)

    return format_string.format(first, second, last)

def gen_email():
    format_string = '{} -X- -X- B-EMAIL\n@ -X- -X- I-EMAIL\n{} -X- -X- E-EMAIL'
    return format_string.format(exrex.getone(r'([a-zA-Z0-9_.-]+)'), exrex.getone(r'([a-z0-9-.]+\.[a-z]{1,5})'))

list = []

# generate phone number
for i in range(300):
    list.append("\n-DOCSTART- -X- -X- O\n\n")

    for j in range(random.randint(0,5)):
        list.append("{}\n".format(gen_phone()))
    for j in range(random.randint(0,5)):
        list.append("{}\n".format(gen_sin()))
    for j in range(random.randint(0,5)):
        list.append("{}\n".format(gen_email()))

file = open('data/test.txt', 'w')
file.writelines(list)
file.close()

print(list)