import sqlglot
import sqlglot.expressions as sql_exp

def get_aggr_ops():
    aggr_type_lst = [
        sql_exp.Sum,
        sql_exp.Count,
        sql_exp.Max,
        sql_exp.Min,
        sql_exp.Avg
    ]
    return aggr_type_lst

def get_select(stmt):
    column_lst = []
    expr_lst = stmt.args.get('expressions', [])
    aggr_type_lst = get_aggr_ops()
    for expr_wrapper in expr_lst:
        if type(expr_wrapper) is sql_exp.Alias:
            expr = expr_wrapper.this
        else:
            expr = expr_wrapper
        col_name = None
        aggr_op = None
        if type(expr) is sql_exp.Column:
            col_name = expr.name
        elif type(expr) in aggr_type_lst:
            aggr_op = expr.__class__.__name__.lower()
            if type(expr.this) is sql_exp.Column:
                col_name = expr.this.name
            elif type(expr.this) is sql_exp.Distinct:
                aggr_op += '_distinct'
                col_name = expr.this.expressions[0].name

        if (col_name is not None) or (aggr_op is not None): 
            col_info = {}
            if col_name is not None:
                col_info['col_name'] = col_name
            if aggr_op is not None:
                col_info['aggr'] = aggr_op
            column_lst.append(col_info)
    return column_lst

def get_op_types():
    op_types = (
        sql_exp.Between,
        sql_exp.EQ,
        sql_exp.GT,
        sql_exp.GTE,
        sql_exp.LT,
        sql_exp.LTE,
        sql_exp.Is,
        sql_exp.Like,
        sql_exp.In,
        sql_exp.NEQ
    )
    return op_types

def get_where(stmt):
    cond_lst = []
    op_types = get_op_types()
    where_node = stmt.args.get('where', None)
    if where_node is None:
        return cond_lst
    cond_expr_generator = where_node.find_all(op_types)
    for cond_expr in cond_expr_generator:
        col_name = None
        op = None
        val = None
        val_2 = None
        expected = True
        if type(cond_expr) is sql_exp.Between:
            #import pdb; pdb.set_trace()
            col_name = cond_expr.this.name if type(cond_expr.this) == sql_exp.Column else ''
            op = 'between'
            val = cond_expr.args['low'].name
            val_2 = cond_expr.args['high'].name

        elif type(cond_expr) is sql_exp.EQ:
            col_name = cond_expr.this.name if type(cond_expr.this) == sql_exp.Column else ''
            op = '='
            val = cond_expr.expression.name
        elif type(cond_expr) is sql_exp.GT:
            col_name = cond_expr.this.name if type(cond_expr.this) == sql_exp.Column else ''
            op = '>'
            val = cond_expr.expression.name
        elif type(cond_expr) is sql_exp.GTE:
            col_name = cond_expr.this.name if type(cond_expr.this) == sql_exp.Column else ''
            op = '>='
            val = cond_expr.expression.name
        elif type(cond_expr) is sql_exp.LT:
            col_name = cond_expr.this.name if type(cond_expr.this) == sql_exp.Column else ''
            op = '<'
            val = cond_expr.expression.name
        elif type(cond_expr) is sql_exp.LTE:
            col_name = cond_expr.this.name if type(cond_expr.this) == sql_exp.Column else ''
            op = '<='
            val = cond_expr.expression.name
        elif type(cond_expr) is sql_exp.Is:
            col_name = cond_expr.this.name if type(cond_expr.this) == sql_exp.Column else ''
            op = 'is'
            val = cond_expr.expression.name
        elif type(cond_expr) is sql_exp.Like:
            col_name = cond_expr.this.name if type(cond_expr.this) == sql_exp.Column else ''
            op = 'like'
            val = cond_expr.expression.name
        elif type(cond_expr) is sql_exp.In:
            col_name = cond_expr.this.name if type(cond_expr.this) == sql_exp.Column else ''
            op = 'in'
            val = [a.name for a in cond_expr.expressions]
        elif type(cond_expr) is sql_exp.NEQ:
            col_name = cond_expr.this.name if type(cond_expr.this) == sql_exp.Column else ''
            op = '!='
            val = cond_expr.expression.name
        else:
            expected = False
        if col_name is None or col_name == '':
            expected = False
        if expected:
            cond_info = {
                'col_name':col_name,
                'op':op,
                'val':val
            }
            if type(cond_expr) is sql_exp.Between:
                cond_info['val_2'] = val_2
            cond_lst.append(cond_info)
    return cond_lst

def get_group_by(stmt):
    group_node = stmt.args.get('group', None)
    if group_node is None:
        return None
    
    col_node = group_node.find(sql_exp.Column)
    if col_node is None:
        return None
    col_name  = col_node.name
    group_by = {
        'col_name':col_name
    }
    return group_by

def get_having(stmt):
    having_node = stmt.args.get('having', None)
    if having_node is None:
        return None
    
    op_types = get_op_types()
    cond_expr = having_node.this
    col_name = None
    aggr_op = None
    pred_op = None    
    val = None
    val_2 = None
    aggr_op_all = get_aggr_ops()
    if type(cond_expr) in (sql_exp.Between, sql_exp.GT, sql_exp.LT):
        if type(cond_expr) is sql_exp.Between:
            pred_op = 'between'
            val = cond_expr.args['low'].name
            val_2 = cond_expr.args['high'].name
        elif type(cond_expr) is sql_exp.GT:
            pred_op = '>'
            val = cond_expr.expression.name
        else:
            pred_op = '<'
            val = cond_expr.expression.name

        aggr_node = cond_expr.this
        if type(aggr_node) in aggr_op_all:
            aggr_op = aggr_node.sql_name().lower()
            col_name = aggr_node.this.name
            cond_info = {
                'col_name':col_name,
                'aggr':aggr_op,
                'pred_op':pred_op,
                'val':val
            }
            if pred_op == 'between':
                cond_info['val_2'] = val_2
            return cond_info
    
    return None

def get_order_by(stmt):
    order_node = stmt.args.get('order', None)
    if order_node is None:
        return None
    col_node = order_node.expressions[0].find(sql_exp.Column)
    if col_node is None:
        return None
    col_name = col_node.this.name
    direction = 'desc' if order_node.expressions[0].args['desc'] else 'asc'
    order_by = {
        'col_name':col_name,
        'direction':direction
    }
    return order_by

def get_limit(stmt):
    limit_node = stmt.args.get('limit', None)
    if limit_node is None:
        return None
    top = int(limit_node.expression.this)
    limit = {
        'top':top
    }
    return limit

def parse_sql(sql):
    return sqlglot.parse_one(sql)
