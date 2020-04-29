import sys
import argparse
from pyspark.sql import SparkSession

from os import listdir
from os.path import isfile, join
import os
import warnings
import datetime
from datetime import date, timedelta, datetime

from brand_lterm_preference.FModel import FModel



def define_argparser():
    parser = argparse.ArgumentParser('yeondys_daily_brand_etl')

    # parser.add_argument('--types', help='all or aday')
    # parser.add_argument('--out-cb-tbl', help='output table for cust-brand')
    # parser.add_argument('--out-brand-tbl', help='output table for brand')
    # parser.add_argument('--out-cc-tbl', help='output table for cust-cate')
    # parser.add_argument('--start-date', help='start_date for etl')
    # parser.add_argument('--feat-cust-tbl', help='feat table for cust')
    # parser.add_argument('--feat-brand-tbl', help='feat table for brand')
    parser.add_argument('--target-date', help='target date')

    args = parser.parse_args()
    return (parser, args)




def main():
    spark_session = SparkSession.builder \
        .appName('Yeondys') \
        .config("spark.driver.memory", "15g") \
        .enableHiveSupport() \
        .getOrCreate()



    formatted_target_date = datetime.strptime(args.target_date, '%Y-%m-%d')
    begin_date = formatted_target_date - timedelta(days=int(6))
    end_date = formatted_target_date + timedelta(days=int(1))
    begin_date = begin_date.strftime('%Y%m%d')
    end_date = end_date.strftime('%Y%m%d')
    print("formatted_target_date : {}, begin_date : {}, end_date : {}".format(formatted_target_date, begin_date, end_date) )
    #
    # date_list = date_generator(begin_date, end_date)
    # str_date_list = "('" + "','".join(date_list) + "')"
    #
    # print("date list : {}".format(str_date_list))
    # print("end_date : {}".format(add_quote(end_date)))
    # print("output_table : {}".format(add_quote(args.output_table)))

    model = FModel()
    model.make_recall()
    model.make_brand_index()
    model.make_pcid_feat()
    model.make_brand_feat()
    model.make_train_validate_testset()
    model.create_model()
    model.sterm_reranking()



def test():
    print(os.getcwd())
    print(os.listdir(os.getcwd()))
    #os.system('xlearn/build/xlearn_predict tmp_test.txt /opt/download/users/22934/model.out')
    #os.system('cp tmp_test.txt /opt/download/users/22934')

    cmd = """
    /home/brand_lterm_preference/xlearn/build/xlearn_predict /home/brand_lterm_preference/tmp_test.txt /opt/download/users/22934/model.out -o /opt/download/users/22934/test.out
    """
    print(cmd)
    os.system(cmd)

    print(os.listdir(os.getcwd()))

    model_path = '/opt/download/users/22934'
    # model_path2 = 'xlearn/build'
    #
    onlyfiles = os.listdir(model_path)
    # onlyfiles2 = os.listdir(model_path2)
    print("dddid")
    print(onlyfiles)


    # # os.system('xlearn/build/xlearn_predict tmp_test.txt /opt/download/users/22934/model.out -o ')
    # df = pd.read_csv('tmp_test.txt.out')
    # print(df.head())

    # # print(onlyfiles)
    # if os.path.isfile('xlearn/build/xlearn_predict'):
	#     print("true")
    # else:
    #     print("false")


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser, args = define_argparser()
    #test()
    main()
