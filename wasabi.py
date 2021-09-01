import json
import numpy as np

# colour codes
# neoplastic - black,
# inflammatory - yellow,
# conective - pink
# necrotic - green
# epithelial - red
# background - blue

viz_info = {
    "line_width": 3,
    "type_colour": {
        0: [0, 0, 0, 0],
        1: [255, 255, 0, 0],
        2: [255, 0, 153, 0],
        3: [0, 255, 0, 0],
        4: [255, 0, 0, 0],
        5: [0, 255, 255, 0],
    },
    "type_name": {
        0: "Neoplastic",
        1: "Inflammatory",
        2: "Connective",
        3: "Necrotic",
        4: "Epithelial",
        5: "Background"
    }
}


def to_wasabi(save_path, inst_info_dict, viz_info, mode, scale_factor, annotator):

    line_width = viz_info["line_width"]

    ann_list_all = []
    type_list_all = []
    for idx, inst_info in inst_info_dict['nuc'].items():
        ann_list_all.append(inst_info[mode])
        if "type" in inst_info.keys():
            type_list_all.append(inst_info["type"])
        else:
            type_list_all.append(-1)

    def gen_wasabi_dict(id, coords, type_name, type_color, mode, line_width, different_colour=False):
        new_dict = {
            "fillColor": "rgba({0},{1},{2},{3})".format(*type_color),
            "id": "{:024d}".format(id),
            "label": {"value": "nuclei"},
            "group": type_name,
        }
        if mode == "centroid":
            if different_colour:
                update_dict = {
                    "lineColor": "rgb({0},{1},{2})".format(*type_color),
                    "type": "point",
                    "center": coords,
                    "lineWidth": line_width
                }
            else:
                update_dict = {
                    "lineColor": "rgb(0, 0, 0)",
                    "type": "point",
                    "center": coords,
                    "lineWidth": line_width,
                }
            new_dict.update(update_dict)
        elif mode == "contour":
            update_dict = {
                "lineColor": "rgb({0},{1},{2})".format(*type_color),
                "type": "polyline",
                "closed": True,
                "points": coords,
                "lineWidth": line_width,
            }
            new_dict.update(update_dict)

        return new_dict

    format_obj_list = []
    for i, ann in enumerate(ann_list_all):
        lab = type_list_all[i]
        if mode == "contour":
            pts_list = np.ceil(np.array(ann) * scale_factor)
            pts_list = [[int(v[0]), int(v[1]), 0] for v in pts_list]
        elif mode == "centroid":
            pos = ann * scale_factor
            pts_list = [int(pos[0]), int(pos[1]), 0]

        type_name = viz_info["type_name"][lab]
        type_colour = viz_info["type_colour"][lab]
        obj_dict = gen_wasabi_dict(i, pts_list, type_name, type_colour, mode, line_width, different_colour=True)
        format_obj_list.append(obj_dict)
    output_dict = {
        "annotation": {
            "description": "",
            "elements": format_obj_list,
            "name": annotator,
        }
    }
    with open(save_path, "w") as handle:
        json.dump(output_dict, handle)
    return


wasabi_path = (
    "/data/Test/hoverNetUHCWTest/json/H10-3166_B2H_and_E_1_first5k_wasabi.json"
)
json_to_convert_str = (
    "/data/Test/hoverNetUHCWTest/json/H10-3166_B2H_and_E_1_first5000.json"
)

with open(
    "/data/Test/hoverNetUHCWTest/json/H10-3166_B2H_and_E_1_first5000.json", "r"
) as myfile:
    data = myfile.read()

json_to_convert = json.loads(data)

read_scale_factor = 1

to_wasabi(
    wasabi_path,
    json_to_convert,
    viz_info,
    "centroid",
    read_scale_factor,
    "panNuke_hovernet_5k",
)
