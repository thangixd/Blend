import sql_parser

def main():
    sql = 'select product, count(distinct quantity) from product_sale where sale_date between "2024-01-01" and "2024-06-30" and catgeory = "Beauty" and store_location = "New York"  group by product having sum(quanlity) > 10 order by quanlity desc limit 3 '

    sql_2 = ' \nSELECT "borough", "cohort", "local_#", "cohort_#"\nFROM table_name\nWHERE "tasc_(ged)_#" = 20 AND "sacc_(iep_diploma)_#" = 60\nORDER BY "advanced_regents_#" DESC\nLIMIT 5;\n'
    
    sql_3 = 'SELECT SUM("cohort_#")  FROM table_name  WHERE cohort_year = 2013'

    sql_4 = 'select max("total_regents_%_of_grads") as "max_total_regents_percentage", max("tasc_(ged)_%_of_cohort") as "max_ged_percentage" from "2001- 2013 Graduation Outcomes Borough- ALL STUDENTS,SWD,GENDER,ELL,ETHNICITY,EVER ELL" where "dropout_#" between 2618 and 3724'

    sql_5 = "SELECT movie_release_year, director_name FROM movies WHERE movie_release_year IS NOT NULL ORDER BY movie_release_year ASC LIMIT 1"

    sql_6 = "SELECT COUNT(email) FROM client WHERE email NOT LIKE '%@gmail.com'"

    sql_7 = "SELECT train_id FROM cars WHERE shape IN ('elipse', 'bucket') GROUP BY train_id"
    sql_8 = "SELECT COUNT(outcome) FROM callcenterlogs WHERE outcome != 'AGENT'"

    sql_9 = 'SELECT movie_title FROM movies WHERE movie_release_year = 1945 ORDER BY movie_popularity DESC LIMIT 1'

    sql_10 = 'SELECT COUNT(*) FROM lists WHERE SUBSTR(list_update_timestamp_utc, 1, 4) - SUBSTR(list_creation_timestamp_utc, 1, 4) > 10'
    
    sql = "SELECT list_title FROM lists WHERE strftime('%Y', list_update_timestamp_utc) = '2016' ORDER BY list_update_timestamp_utc DESC LIMIT 1"
    
    sql = "SELECT course_id FROM taughtBy WHERE course_id = 11 OR course_id = 18 GROUP BY course_id ORDER BY COUNT(course_id) DESC LIMIT 1"

    sql = "SELECT STRFTIME('%Y', t1.paymentDate), COUNT(t1.customerNumber) FROM payments AS t1 WHERE t1.amount < 10000 GROUP BY STRFTIME('%Y', t1.paymentDate)"
    
    sql = "SELECT App FROM playstore WHERE Price = 0 ORDER BY CAST(REPLACE(REPLACE(Installs, ',', ''), '+', '') AS INTEGER) DESC LIMIT 5"

    sql = "SELECT App FROM playstore WHERE Price = 0 ORDER BY Installs DESC LIMIT 5"

    stmt = sql_parser.parse_sql(sql)
    import pdb; pdb.set_trace()
    # stmt.args: 'expressions', 'where', 'group', 'having', 'order', 'limit',
    print("select  ", sql_parser.get_select(stmt))
    print("where  ", sql_parser.get_where(stmt))
    print("group by ", sql_parser.get_group_by(stmt))
    print("having ", sql_parser.get_having(stmt))
    print("order by", sql_parser.get_order_by(stmt))
    print("limit", sql_parser.get_limit(stmt))

if __name__ == '__main__':
    main()