from lenses import lens
import json

#data json from list of lists of dicts to list of dicts
def data_convert(data):
    data_new=[]
    for i in range(len(data)):
        dict_new={}
        for j in range(len(data[i])):
            
            key=list(data[i][j].keys())[0]
            value=list(data[i][j].values())[0]
            dict_new.update({key:value})
        data_new.append(dict_new)
    return data_new


#Get menu infomation
def get_menu(data_new):
    menu_true={"description":[],
    "unit_price":[],
    "quantity":[],
    "subtotal":[],
    "vat_rate":[]}
    menu_false={"description":[],
    "unit_price":[],
    "quantity":[],
    "subtotal":[],
    "vat_rate":[]}
    for i in range(len(data_new)):
        if data_new[i].get("menu.description") is None:
            continue
        try:
            unit_price=eval(data_new[i].get("menu.unit_price").replace(".","").replace(",","."))
            quantity=eval(data_new[i].get("menu.quantity"))
            subtotal=eval(data_new[i].get("menu.total").replace(".","").replace(",","."))
            if subtotal==unit_price*quantity:
                menu_true["description"].append(data_new[i].get("menu.description"))
                menu_true["unit_price"].append(unit_price)
                menu_true["quantity"].append(quantity)
                menu_true["subtotal"].append(subtotal)
            else:
                menu_false["description"].append(data_new[i].get("menu.description"))
                menu_false["unit_price"].append(unit_price)
                menu_false["quantity"].append(quantity)
                menu_false["subtotal"].append(subtotal)
        except:
            menu_false["description"].append(data_new[i].get("menu.description"))
            menu_false["unit_price"].append(data_new[i].get("menu.unit_price"))
            menu_false["quantity"].append(data_new[i].get("menu.quantity"))
            menu_false["subtotal"].append(data_new[i].get("menu.total"))


    menu_vat=[]
    menu_vat_sign=False
    for i in range(len(data_new)):
        if data_new[i].get("total.vat_rate") is not None:
            vat_rate= eval(data_new[i].get("total.vat_rate").replace("%",""))
            menu_true.update({"vat_rate":vat_rate})
        if data_new[i].get("menu.vat_rate") is not None:
            menu_vat_sign=True
            vat_rate= eval(data_new[i].get("total.vat_rate").replace("%",""))
            menu_vat.append(vat_rate)
    if menu_vat_sign==True:
        menu_true["vat_rate"]=menu_vat
    return menu_true, menu_false

#Get info, seller, customer
def get_main_info(data_new):
    main_info={}
    key_list=['info.date','info.sign_date','info.form','info.serial','info.num','seller.company','seller.tax','seller.address','customer.company','customer.tax','customer.address']
    # lens.Each().Get('info.serial', default=None).Instance(str).get()(data_new) 
    for key in key_list:
        main_info[key]=None
    for key in key_list:
        for data in data_new:
            if main_info[key]==None:
                main_info[key]=lens.Get(key, default=None).get()(data)
    return main_info


#Get total for service invoice
def get_total(data_new):
    total_true={}
    total_false={}
    total_sign=False
    for i in range(len(data_new)):
        if data_new[i].get("total.total") is None:
            continue
        try:
            subtotal=eval(data_new[i].get("total.subtotal").replace(".","").replace(",","."))
            vat_rate= eval(data_new[i].get("total.vat_rate").replace("%",""))
            vat=eval(data_new[i].get("total.vat"))
            total=eval(data_new[i].get("total.total").replace(".","").replace(",","."))
            
            if total==subtotal+vat:
                total_true["total.subtotal"]=subtotal
                total_true["total.vat_rate"]=vat_rate
                total_true["total.vat"]=vat
                total_true["total.total"]=total
                total_sign=True
        except:
            total_false["total.subtotal"]=data_new[i].get("total.subtotal")
            total_false["total.vat_rate"]=data_new[i].get("total.vat_rate")
            total_false["total.vat"]=data_new[i].get("total.vat")
            total_false["total.total"]=data_new[i].get("total.total")
    if total_sign:
        return total_true,total_sign
    return total_false,total_sign