from __future__ import absolute_import, division, print_function

import math
import os
import random

import exrex
from openpyxl import load_workbook
from tokenizer import Tokenizer

tokenizer = Tokenizer('en')


def convert_bio_to_bioes(file, newfile):
    with open(file, encoding='UTF-8') as f:
        file_content = f.readlines()

    new_file_content = []

    for line, next_line in zip(file_content, file_content[1:] + ['']):
        line = line.strip()
        next_line = next_line.strip()

        if len(line) == 0 or line.startswith("-DOCSTART-"):
            new_file_content.append(line)
        else:
            l = line.split(' ')
            nl = next_line.split(' ')

            word, tag, next_tag = l[0], l[-1], nl[-1]
            tag1 = tag.split('-')[0]
            next_tag1 = next_tag.split('-')[0]

            if tag1 in ['B', 'I']:
                if tag1 == 'B':
                    tag1 = 'S' if next_tag1 not in ['I', 'E'] else tag1
                if tag1 == 'I':
                    tag1 = 'E' if next_tag1 not in ['I', 'E'] else tag1

                new_file_content.append(word + ' ' + tag1 + '-' + tag.split('-')[1])
            else:
                new_file_content.append(word + ' ' + tag)

    with open(newfile, mode="w", encoding='UTF-8') as f:
        for i, line in enumerate(new_file_content):
            f.write("{}\n".format(line))


def convert_xlsx_and_split(file, split=0.7):
    wb = load_workbook(file, guess_types=False)
    split_idx = math.floor(wb.active.max_row * split)
    train_file = os.path.dirname(file) + '/train.txt'
    valid_file = os.path.dirname(file) + '/valid.txt'

    with open(train_file, mode='w', encoding='UTF-8') as train_f:
        with open(valid_file, mode='w', encoding='UTF-8') as valid_f:

            header = None
            header_mask = None
            for idx, row in enumerate(wb.active.values):
                if header is None:
                    header = [x.replace(' ', '') for x in row if x is not None]
                    header_mask = [False if x.lower() in get_disabled_col() else True for x in header]
                    continue

                f = train_f if idx < split_idx else valid_f
                row_string = []
                for i, cell in enumerate(row):
                    if i < len(header) and header_mask[i]:
                        row_string.append(str(cell).strip())

                for i, doc in enumerate(tokenizer.split(row_string)):
                    word_raw = [token.text for token in doc if not token.is_space]
                    for j, word in enumerate(word_raw):
                        f.write('{}\n'.format(
                            word + ' ' + header[i] + ' ' + (
                                'O' if cell is None or str(cell).lower() in ['na', 'not applicable'] else get_tag(
                                    header[i], len(word_raw), j))))

                f.write('\n')


def get_tag(field_name, word_list_length, word_index):
    if field_name.lower() in ['fullname', 'firstname', 'middlename', 'lastname', 'contact-lastname',
                              'contact-firstname', 'contactname', 'principalname', 'principalfirstname',
                              'principallastname', 'fundcontact']:
        return get_tag1(word_list_length, word_index) + '-' + 'PERSON'
    # elif field_name.lower() in ['businessname', 'organization', 'localname', 'company', 'schoolname']:
    #     return get_tag1(word_list_length, word_index) + '-' + 'ORG'
    elif field_name.lower() in ['telephonenumber', 'telephone', 'phone', 'phonenumber', 'faxnumber', 'fax', 'phone1',
                                'phone2', 'phonefax', 'phonetollfree']:
        return get_tag1(word_list_length, word_index) + '-' + 'PHONE'
    elif field_name.lower() in ['emailaddress', 'email', 'e-mail']:
        return get_tag1(word_list_length, word_index) + '-' + 'EMAIL'
    elif field_name.lower() in ['streetaddress', 'postalcode', 'address', 'street', 'physicaladdress',
                                'buildingaddress']:
        return get_tag1(word_list_length, word_index) + '-' + 'ADDRESS'
    elif field_name.lower() in ['postalcode']:
        return get_tag1(word_list_length, word_index) + '-' + 'POSTALCODE'
    else:
        return 'O'


def get_tag1(word_list_length, word_index):
    if word_list_length == 1:
        tag1 = 'S'
    elif word_index == 0:
        tag1 = 'B'
    elif word_index < word_list_length - 1:
        tag1 = 'I'
    else:
        tag1 = 'E'

    return tag1


def get_disabled_col():
    return ['description', 'areasofspecialization', 'professionalassociation']


def gen_phone():
    format1 = '{} -X- -X- B-PHONE\n- -X- -X- I-PHONE\n{} -X- -X- I-PHONE\n- -X- -X- I-PHONE\n{} -X- -X- E-PHONE\n'  # 123-456-7890
    format4 = '{} -X- -X- B-PHONE\n{} -X- -X- I-PHONE\n{} -X- -X- E-PHONE\n'  # 123 456 7890
    format2 = '( -X- -X- B-PHONE\n{} -X- -X- I-PHONE\n) -X- -X- I-PHONE\n{} -X- -X- I-PHONE\n- -X- -X- I-PHONE\n{} -X- -X- E-PHONE\n'  # (123) 456-7890
    format3 = '( -X- -X- B-PHONE\n{}){} -X- -X- I-PHONE\n- -X- -X- I-PHONE\n{} -X- -X- E-PHONE\n'  # (123)456-7890
    format5 = '{} -X- -X- B-PHONE\n. -X- -X- I-PHONE\n{} -X- -X- I-PHONE\n. -X- -X- I-PHONE\n{} -X- -X- E-PHONE\n'  # 123.456.7890

    formatDict = {1: format1, 2: format2, 3: format3, 4: format4, 5: format5}

    first = str(random.randint(100, 999))
    second = str(random.randint(1, 999)).zfill(3)
    last = str(random.randint(1, 9999)).zfill(4)

    return formatDict[random.randint(1, 4)].format(first, second, last)


def gen_sin():
    format_string = '{} -X- -X- B-SIN\n{} -X- -X- I-SIN\n{} -X- -X- E-SIN\n'  # 123 456 789

    first = str(random.randint(1, 999)).zfill(3)
    second = str(random.randint(1, 999)).zfill(3)
    last = str(random.randint(1, 999)).zfill(3)

    return format_string.format(first, second, last)


def gen_email():
    format_string = '{} -X- -X- B-EMAIL\n@ -X- -X- I-EMAIL\n{} -X- -X- E-EMAIL'
    return format_string.format(exrex.getone(r'([a-zA-Z0-9_.-]+)'), exrex.getone(r'([a-z0-9-.]+\.[a-z]{1,5})'))


def generate_fake_email_phone_sin():
    list = []
    for i in range(300):
        list.append("\n-DOCSTART- -X- -X- O\n\n")

        for j in range(random.randint(0, 5)):
            list.append("{}\n".format(gen_phone()))
        for j in range(random.randint(0, 5)):
            list.append("{}\n".format(gen_sin()))
        for j in range(random.randint(0, 5)):
            list.append("{}\n".format(gen_email()))

    with open('../data/raw/gen/train.txt', 'w') as f:
        f.writelines(list)

    list.clear()
    for i in range(300):
        list.append("\n-DOCSTART- -X- -X- O\n\n")

        for j in range(random.randint(0, 5)):
            list.append("{}\n".format(gen_phone()))
        for j in range(random.randint(0, 5)):
            list.append("{}\n".format(gen_sin()))
        for j in range(random.randint(0, 5)):
            list.append("{}\n".format(gen_email()))

    with open('../data/raw/gen/valid.txt', 'w') as f:
        f.writelines(list)


if __name__ == '__main__':
    # convert_bio_to_bioes('../data/raw/conll2003/en/train.txt', '../data/raw/conll2003/en/train_bioes.txt')
    # convert_bio_to_bioes('../data/raw/conll2003/en/test_bio.txt', '../data/raw/conll2003/en/test_bioes.txt')
    # convert_bio_to_bioes('../data/raw/conll2003/en/valid.txt', '../data/raw/conll2003/en/valid_bioes.txt')

    convert_xlsx_and_split('../data/raw/City_biz_incubator/BusinessEcosystem.xlsx')
    convert_xlsx_and_split('../data/raw/City_door_open/Doors_Open_2018.xlsx')
    convert_xlsx_and_split('../data/raw/cymh/cymh_mcys_open_data_dataset_may_2016_cyrb_approved.xlsx')
    convert_xlsx_and_split('../data/raw/Farm_marketing_board/marketingdirectory1.xlsx')
    convert_xlsx_and_split('../data/raw/ON_farm_advisor/growing_forward_2_farm_advisors.xlsx')
    convert_xlsx_and_split('../data/raw/ON_labor_sponsored_inv_fund/labour-sponsored-investment-funds_1.xlsx')
    convert_xlsx_and_split('../data/raw/ON_livestock/lma-listc.xlsx')
    convert_xlsx_and_split('../data/raw/ON_private_school/private_schools_contact_information_july_2018_en_.xlsx')
    convert_xlsx_and_split(
        '../data/raw/ON_public_library/2015_ontario_public_library_statistics_open_data_dec_2017rev.xlsx')
    convert_xlsx_and_split('../data/raw/ON_public_school/publicly_funded_schools_xlsx_july_2018_en.xlsx')
    convert_xlsx_and_split('../data/raw/ON_school_board/boards_schoolauthorities_july_2018_en.xlsx')
    convert_xlsx_and_split('../data/raw/Toxics_planner/toxics_planners.xlsx')
    convert_xlsx_and_split('../data/raw/WoodSupplier/may2018.xlsx')

    generate_fake_email_phone_sin()
