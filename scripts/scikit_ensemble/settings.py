from utils.definitions import save_obj

# name = [
#     # cat1
#     ["nlp", "top_pop"],
#     # cat2
#     ["cbf_item_album", "cbf_item_artist", "cbf_user_album","cbf_user_artist", "rp3beta", "cf_user", "nlp", "slim","pers_top_pop", "pers_top_pop_2", "svd", "asl", "cbf_item_album_artist"],
#     # cat3
#     ["cbf_item_album", "cbf_item_artist", "cbf_user_album","cbf_user_artist", "rp3beta", "cf_user", "nlp", "slim", "svd", "asl", "cbf_item_album_artist"],
#     # cat4
#     ["cbf_item_album", "cbf_item_artist", "cbf_user_album","cbf_user_artist", "rp3beta", "cf_user", "slim", "svd", "asl", "cbf_item_album_artist"],
#     # cat5
#     ["cbf_item_album", "cbf_item_artist", "cbf_user_album","cbf_user_artist", "rp3beta", "cf_user", "nlp", "slim", "svd", "asl", "cbf_item_album_artist"],
#     # cat6
#     ["cbf_item_album", "cbf_item_artist", "cbf_user_album","cbf_user_artist", "rp3beta", "cf_user", "slim", "svd", "asl", "cbf_item_album_artist"],
#     # cat7
#     ["cbf_item_album", "cbf_item_artist", "cbf_user_album","cbf_user_artist", "rp3beta", "cf_user", "nlp", "slim", "svd", "asl", "cbf_item_album_artist"],
#     # cat8
#     ["cbf_item_album", "cbf_item_artist", "cbf_user_album","cbf_user_artist", "rp3beta", "cf_user", "nlp", "slim", "svd", "asl", "cbf_item_album_artist"],
#     # cat9
#     ["cbf_item_album", "cbf_item_artist", "cbf_user_album","cbf_user_artist", "rp3beta", "cf_user", "nlp", "slim", "rp3beta_cat9", "svd", "asl", "cbf_item_album_artist"],
#     # cat10
#     ["cbf_item_album", "cbf_item_artist", "cbf_user_album","cbf_user_artist", "rp3beta", "cf_user", "nlp", "slim", "svd", "asl", "cbf_item_album_artist"]]
name = [
    # cat1
    ["nlp", "top_pop"],
    # cat2
    ["cbf_item_album", "cbf_item_artist", "cbf_user_album","cbf_user_artist", "rp3beta", "cf_user", "nlp", "slim","pers_top_pop", "pers_top_pop_2"],
    # cat3
    ["cbf_item_album", "cbf_item_artist", "cbf_user_album","cbf_user_artist", "rp3beta", "cf_user", "nlp", "slim"],
    # cat4
    ["cbf_item_album", "cbf_item_artist", "cbf_user_album","cbf_user_artist", "rp3beta", "cf_user", "slim"],
    # cat5
    ["cbf_item_album", "cbf_item_artist", "cbf_user_album","cbf_user_artist", "rp3beta", "cf_user", "nlp", "slim"],
    # cat6
    ["cbf_item_album", "cbf_item_artist", "cbf_user_album","cbf_user_artist", "rp3beta", "cf_user", "slim"],
    # cat7
    ["cbf_item_album", "cbf_item_artist", "cbf_user_album","cbf_user_artist", "rp3beta", "cf_user", "nlp", "slim"],
    # cat8
    ["cbf_item_album", "cbf_item_artist", "cbf_user_album","cbf_user_artist", "rp3beta", "cf_user", "nlp", "slim"],
    # cat9
    ["cbf_item_album", "cbf_item_artist", "cbf_user_album","cbf_user_artist", "rp3beta", "cf_user", "nlp", "slim", "rp3beta_cat9"],
    # cat10
    ["cbf_item_album", "cbf_item_artist", "cbf_user_album","cbf_user_artist", "rp3beta", "cf_user", "nlp", "slim"]]


save_obj(name, "name")
