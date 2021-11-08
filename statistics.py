import os


def cal_acc(result_dir):
    filenames = os.listdir(result_dir)
    result_dict = {}
    for fn in filenames:
        final_path = os.path.join(result_dir, fn)
        with open(final_path, 'r') as f:
            for line in f.readlines():
                if 'correct / total' in line:
                    tmp = line.split(':')[1]
                    correct = int(tmp.split('/')[0].strip())
                    total = int(tmp.split('/')[1].strip())
                    if 'correct' not in result_dict.keys():
                        result_dict['correct'] = [correct]
                    else:
                        result_dict['correct'].append(correct)
                    if 'total' not in result_dict.keys():
                        result_dict['total'] = [total]
                    else:
                        result_dict['total'].append(total)
                if 'macro_f1' in line:
                    f_score = float(line.split(':')[1].strip())
                    if 'f1' not in result_dict.keys():
                        result_dict['f1'] = [f_score]
                    else:
                        result_dict['f1'].append(f_score)

    return result_dict


if __name__ == '__main__':
    
    #result_dict = cal_acc('../svm-result/svm-result25')
    #result_dict = cal_acc('datasets/rest/tmp_optimized_result')
    result_dict = cal_acc('datasets/rest/optimal_results/rounds_10')
    #result_dict = cal_acc('datasets/rest/optimal_results/svm-results-k20')
    correct = sum(result_dict['correct'])
    total = sum(result_dict['total'])
    f_scores = result_dict['f1']
    print(f'f_scores: {f_scores}')
    f1 = 0
    for num_sample, chunk_f in zip(result_dict['total'], f_scores):
        #f1 += num_sample / 1120 * chunk_f
        f1 += num_sample / total * chunk_f
    print('correct / total: %d / %d' % (correct, total))
    print('Acc: %.5f' % (correct / total))
    print('F1: %.5f' % f1)
