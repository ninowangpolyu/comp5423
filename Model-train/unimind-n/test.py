import json

length_list = []


def count_user_in_conversations(json_data):
    conversation = json_data['conversation']
    user_count = sum(1 for line in conversation if line.startswith("user:"))
    return user_count


def count_users_in_file(file_path):
    user_counts = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            json_data = json.loads(line)
            user_count = count_user_in_conversations(json_data)
            user_counts.append(user_count)
    return user_counts


def clear_result():
    test_file = count_users_in_file('./data/test.txt')
    input_file = './data/test_2.txt'
    with open(input_file, encoding='utf8') as fp:
        for line in fp:
            data = json.loads(line)
            conversation = data['conversation']
            goal = data['goal']
            length = len(conversation)
            if goal.startswith("[1] 寒暄"):
                length -= 1
            length_list.append(length)

    for i in range(len(length_list)):
        length_list[i] /= 2
        length_list[i] = int(length_list[i])

    with open('./output/durecdial/durecdial_resp.decoded', 'r') as file:
        # 读取文件所有行
        lines = file.readlines()

    output = []

    line_cout = 0
    for i in range(2626):
        line = test_file[i] - 1 + line_cout
        line_output = lines[line]
        output.append(line_output)
        line_cout += length_list[i]

    # 打开文件
    with open('result.txt', 'w') as file:
        for item in output:
            # 写入文件
            file.write('%s' % item)
