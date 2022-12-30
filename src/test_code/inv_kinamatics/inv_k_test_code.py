#%%
# m1 = [['cos(t1)', '-sin(t1)cos(t1)', 'sin(t1)sin(a1)'], ['sin(t1)', 'cos(t1)cos(a1)', '-cos(t1)sin(a1)'], ['0', 'sin(a1)', 'cos(a1)']]
# m2 = [['cos(t2)', '-sin(t2)cos(t2)', 'sin(t2)sin(a2)'], ['sin(t2)', 'cos(t2)cos(a2)', '-cos(t2)sin(a2)'], ['0', 'sin(a2)', 'cos(a2)']]
# m3 = [['cos(t3)', '-sin(t3)cos(t3)', 'sin(t3)sin(a3)'], ['sin(t3)', 'cos(t3)cos(a3)', '-cos(t3)sin(a3)'], ['0', 'sin(a3)', 'cos(a3)']]

# m1 = [
#     ["cos(t1)", "-sin(t1) * cos(t1)", "-sin(t1)"],
#     ["sin(t1)", "0", "cos(t1)"],
#     ["0", "-1", "0"],
# ]
# m2 = [
#     ["cos(t2)", "-sin(t2) * cos(t2)", "sin(t2)"],
#     ["sin(t2)", "0", "-cos(t2)"],
#     ["0", "1", "0"],
# ]
# m3 = [
#     ["cos(t3)", "-sin(t3) * cos(t3)", "0"],
#     ["sin(t3)", "cos(t3)", "0"],
#     ["0", "0", "1"],
# ]

# f1 = m1
# f2 = m2


# def string_dot_matrix(f1, f2):
#     full = [["", "", ""], ["", "", ""], ["", "", ""]]

#     for x1 in range(3):
#         for x2 in range(3):
#             multiple_add_list = ["", "", ""]
#             for y in range(3):
#                 multiply = [f1[x1][y], f2[y][x2]]
#                 if "0" in multiply:
#                     multiple_add_list[y] = "0"
#                 else:
#                     multiply = [x for x in multiply if x != "1"]
#                     if len(multiply) == 0:
#                         multiply = "1"
#                     else:
#                         multiply = " * ".join(multiply)
#                         multiple_add_list[y] = multiply
#             multiple_add_list = [x for x in multiple_add_list if x != "0"]
#             if len(multiple_add_list) == 0:
#                 multiple_add_list = "0"
#                 full[x1][x2] = multiple_add_list
#             else:
#                 multiple_add_list = " + ".join(multiple_add_list)

#                 full[x1][x2] = "(" + multiple_add_list + ")"

#     return full


# m12 = string_dot_matrix(m1, m2)
# m123 = string_dot_matrix(m12, m3)
# m12

# #%%
# for deg in [-90, 90, 0]:
#     print(np.cos(np.radians(deg)), np.sin(np.radians(deg)))


# [['((cos(t1) * cos(t2) + -sin(t1)cos(t1) * sin(t2)) * cos(t3) + (cos(t1) * -sin(t2)cos(t2) + -sin(t1)) * sin(t3))',
#   '((cos(t1) * cos(t2) + -sin(t1)cos(t1) * sin(t2)) * -sin(t3)cos(t3) + (cos(t1) * -sin(t2)cos(t2) + -sin(t1)) * cos(t3))',
#   '((cos(t1) * sin(t2) + -sin(t1)cos(t1) * -cos(t2)))'],
#  ['((sin(t1) * cos(t2)) * cos(t3) + (sin(t1) * -sin(t2)cos(t2) + cos(t1)) * sin(t3))',
#   '((sin(t1) * cos(t2)) * -sin(t3)cos(t3) + (sin(t1) * -sin(t2)cos(t2) + cos(t1)) * cos(t3))',
#   '((sin(t1) * sin(t2)))'],
#  ['((-1 * sin(t2)) * cos(t3))',
#   '((-1 * sin(t2)) * -sin(t3)cos(t3))',
#   '((-1 * -cos(t2)))']]
