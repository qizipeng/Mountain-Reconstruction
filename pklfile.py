import pickle


with open("/DATA/3dDATA/all_tsdf_9/1/test/google9/fragments.pkl", 'rb') as f:
    file = pickle.load(f)
    print(file)

# with open("/DATA/3dDATA/all_tsdf_9/1/test/google1/fragments_bak.pkl", 'rb') as f:
#     file = pickle.load(f)
#     print(file)

print(file[0]["scene"])
infromation = []
for i in range(9):
    infromation.append({
        'scene': "google9",
        'fragment_id': file[i]["fragment_id"],
        'image_ids': file[i]["image_ids"],
        'vol_origin': file[i]["vol_origin"],
        'voxel_size': file[i]["voxel_size"],
    })

with open("/DATA/3dDATA/all_tsdf_9/1/test/google9/fragments2.pkl", 'wb') as w:
    pickle.dump(infromation,w)