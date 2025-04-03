import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd
from swmm_api import read_inp_file
import matplotlib.font_manager as fm
import re
# font = fm.FontProperties(fname=r"C:\Windows\Fonts\times.ttf", size=30)  # 设置字体文件路径和大小
# font1 = fm.FontProperties(fname=r"C:\Windows\Fonts\times.ttf", size=28)  # 设置字体文件路径和大小
# 设置 colorbar 显示值的字体样式和大小

class NetworkAnalysis:
    def find_upstream_branches(self, G, hub_nodes):
        upstream_paths = {node: [] for node in hub_nodes}

        def dfs(node, path, branch_paths):
            if node in hub_nodes and node != start_node:
                branch_paths.append(path)
                return
            if len(list(G.predecessors(node)))==0:
                branch_paths.append(path)
                return
            for pred in G.predecessors(node):
                new_path = path + [(pred, node)]
                if pred in hub_nodes:
                    branch_paths.append(new_path)
                else:
                    dfs(pred, new_path, branch_paths)

        for start_node in hub_nodes:
            branch_paths = []
            for pred in G.predecessors(start_node):
                dfs(pred, [(pred, start_node)], branch_paths)
            upstream_paths[start_node] = branch_paths

        return upstream_paths
class PltChart:
    def __init__(self,swmm_path):
        self.swmm_inp = swmm_path
        pass
    def double_chart(self,preds,trues,rainfall):
        x_range = len(preds)
        fig, ax1 = plt.subplots()

        # 绘制第一个数据系列（yew）
        color1 = 'red'
        color2 = 'blue'
        color3 = 'black'
        ax1.set_xlabel('Time(min)')
        ax1.set_ylabel('Water-level (m)', color=color3)
        ax1.plot(np.arange(x_range), preds, color=color2)
        ax1.plot(np.arange(x_range), trues, color=color1)
        ax1.tick_params(axis='y', labelcolor=color3)


        # 创建第二个坐标轴，并绘制第二个数据系列（温度）
        ax2 = ax1.twinx()
        color = 'green'
        ax2.set_ylabel('Rainfall (mm/h)', color=color)
        ax2.bar(np.arange(x_range), rainfall, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.invert_yaxis()

        # 设置图表标题
        plt.title('Monthly Rainfall and Temperature')

        # 显示图表
        plt.show()
    # def nx_plot(self,label_nodes,bad_nodes,swmm_inp,save_dir=None):
    #     import networkx as nx
    #     from swmm_api import read_inp_file
    #     import matplotlib.pyplot as plt
    #     inp = read_inp_file(swmm_inp)
    #     coordinates = inp.COORDINATES
    #     nodes = inp.JUNCTIONS
    #     conduits = inp.CONDUITS
    #     outfalls = inp.OUTFALLS
    #     storages = inp.STORAGE
    #     X = nx.DiGraph()
    #     for node in nodes:
    #         X.add_node(node, pos=(coordinates[node]['x'], coordinates[node]['y']))
    #     for outfall in outfalls:
    #         X.add_node(outfall, pos=(coordinates[outfall]['x'], coordinates[outfall]['y']))
    #     for storage in storages:
    #         X.add_node(storage, pos=(coordinates[storage]['x'], coordinates[storage]['y']))
    #     for cunduit in conduits:
    #         inlet = conduits[cunduit]['FromNode']
    #         outlet = conduits[cunduit]['ToNode']
    #         X.add_edge(inlet, outlet)
    #     pos = nx.get_node_attributes(X, 'pos')
    #     node_ids = list(coordinates.keys())
    #     unmask_node = {f'{node_ids[i]}' for i in label_nodes}
    #     poor_node = {f'{node_ids[i]}' for i in bad_nodes}
    #     node_colors = ['red' if node in unmask_node else 'blue' if node in poor_node else '#545454' for node in X.nodes()]
    #     plt.figure(figsize=(10, 10))
    #     nx.draw(X, pos, with_labels=False, node_size=50, node_color=node_colors, font_size=8, font_color='black')
    #     plt.title('SWMM Network Graph')
    #     plt.xlabel('X Coordinate')
    #     plt.ylabel('Y Coordinate')
    #     if save_dir:
    #         plt.savefig(save_dir,dpi=300)
    #         plt.clf()
    #     else:
    #         plt.show()
    def nx_plot(self,label_nodes,nse,swmm_inp,number,save_dir=None):
        from matplotlib.colors import LinearSegmentedColormap,to_rgba
        nse[nse<0]=-0.1
        inp = read_inp_file(swmm_inp)
        coordinates = inp.COORDINATES
        nodes = inp.JUNCTIONS
        conduits = inp.CONDUITS
        outfalls = inp.OUTFALLS
        storages = inp.STORAGE
        X = nx.DiGraph()
        for node in nodes:
            X.add_node(node, pos=(coordinates[node]['x'], coordinates[node]['y']))
        for outfall in outfalls:
            X.add_node(outfall, pos=(coordinates[outfall]['x'], coordinates[outfall]['y']))
        for storage in storages:
            X.add_node(storage, pos=(coordinates[storage]['x'], coordinates[storage]['y']))
        for cunduit in conduits:
            inlet = conduits[cunduit]['FromNode']
            outlet = conduits[cunduit]['ToNode']
            if cunduit in ['JTN-03(tb2).1','JTN-03(tb).2']:
                pass
            else:
                X.add_edge(inlet, outlet)
        # degrees = X.degree()
        # nodes_with_edges_than_2 = [id_node for id_node,degree in degrees if degree>2]
        # nodes_with_edges_than_2 = set(nodes_with_edges_than_2)
        # degress_color = ['blue' for _ in range(len(nodes_with_edges_than_2))]

        pos = nx.get_node_attributes(X, 'pos')
        node_ids = np.array(list(coordinates.keys()))

        unmask_node = {f'{node_ids[i]}' for i in label_nodes}
        unmask_color = ['lime' for _ in range(len(unmask_node))]
        parula_map = 'RdYlBu'

        plt.figure(figsize=(11, 11))
        nx.draw(X, pos, with_labels=False, node_size=100, node_color=nse, cmap=parula_map,font_size=8, font_color='black')

        # nx.draw_networkx_edges(X, pos, arrows=False)

        nx.draw_networkx_nodes(X, pos, nodelist=unmask_node, node_shape='*', edgecolors='black',node_size=700, node_color=unmask_color)
        # plt.text(0.08, 0.89, f'({chr(97 + number)})', transform=plt.gca().transAxes,
        #          fontweight='bold', va='top', ha='left')

        sm = plt.cm.ScalarMappable(cmap=parula_map,norm=plt.Normalize(vmin=min(nse),vmax=1))
        sm.set_array([])  # 这是一个技巧，以便colorbar知道要使用的颜色映射
        cbar_ax = plt.gcf().add_axes([0.90, 0.15, 0.02, 0.7])
        cbar = plt.colorbar(sm,cax=cbar_ax)

        tick_values = np.arange(0,1.1,0.1)
        tick_values = np.insert(tick_values,0,-0.1)
        cbar.set_ticks(tick_values)
        cbar.set_ticklabels([f'{val:.1f}' for val in tick_values])

        # plt.xlabel('X Coordinate')
        # plt.ylabel('Y Coordinate')
        if save_dir:
            plt.savefig(save_dir,dpi=300)
            plt.clf()
        else:
            plt.show()
    def transfer_nodeid(self,swmm_ids):
        inp = read_inp_file(self.swmm_inp)
        coordinates = inp.COORDINATES
        node_ids = np.array(list(coordinates.keys()))
        python_nodes = [np.where(node_ids == swmm_id)[0][0] for swmm_id in swmm_ids]
        swmm_id = swmm_ids[0]

        return python_nodes
    def links_branch(self):
        inp = read_inp_file(self.swmm_inp)
        coordinates = inp.COORDINATES
        subcatchments ={inp.SUBCATCHMENTS[sbucatchment]['Outlet']:inp.SUBCATCHMENTS[sbucatchment]['Area'] for sbucatchment in inp.SUBCATCHMENTS}
        nodes = inp.JUNCTIONS
        conduits = inp.CONDUITS
        outfalls = inp.OUTFALLS
        storages = inp.STORAGE
        X = nx.DiGraph()
        for node in nodes:
            X.add_node(node, pos=(coordinates[node]['x'], coordinates[node]['y']))
        for outfall in outfalls:
            X.add_node(outfall, pos=(coordinates[outfall]['x'], coordinates[outfall]['y']))
        for storage in storages:
            X.add_node(storage, pos=(coordinates[storage]['x'], coordinates[storage]['y']))
        for cunduit in conduits:
            inlet = conduits[cunduit]['FromNode']
            outlet = conduits[cunduit]['ToNode']
            if cunduit in ['JTN-03(tb2).1','JTN-03(tb).2']:
                pass
            else:
                X.add_edge(inlet, outlet,length=conduits[cunduit]['Length'],
                           diameter=inp.XSECTIONS[cunduit]['Geom1'] if cunduit in inp.XSECTIONS else 0,
                           slop=(conduits[cunduit]['InOffset']-conduits[cunduit]['OutOffset'])/conduits[cunduit]['Length'])
        node_ids = np.array(list(coordinates.keys()))

        degrees = X.degree()
        nodes_with_edges_than_2 = [id_node for id_node,degree in degrees if degree>2]

        junction_nodes = [np.where(node_ids==junc)[0][0] for junc in nodes_with_edges_than_2]
        network_analysis = NetworkAnalysis()
        upstream_paths = network_analysis.find_upstream_branches(X, nodes_with_edges_than_2)
        links_ = [i for key,value in upstream_paths.items() for i in value]
        links = [i for i in links_ if len(i)>1]
        up_path=[]
        for tuple_list in links:
            extend_list = [t[0] for t in tuple_list]
            extend_list.insert(0,tuple_list[0][1])
            up_path.append(extend_list)
        branch_index = [[np.where(np.array(node_ids)==up_path[s][i])[0][0] for i in range(len(up_path[s]))] for s in range(len(up_path))]
        out_links=[]
        for out_link in branch_index:
            if out_link[-1] not in junction_nodes:
                out_links.append(out_link)
        return branch_index,out_links

    def connection_sys(self,label_nodes,nse,swmm_inp):
        inp = read_inp_file(swmm_inp)
        coordinates = inp.COORDINATES
        subcatchments ={inp.SUBCATCHMENTS[sbucatchment]['Outlet']:inp.SUBCATCHMENTS[sbucatchment]['Area'] for sbucatchment in inp.SUBCATCHMENTS}
        nodes = inp.JUNCTIONS
        conduits = inp.CONDUITS
        outfalls = inp.OUTFALLS
        storages = inp.STORAGE
        X = nx.DiGraph()
        for node in nodes:
            X.add_node(node, pos=(coordinates[node]['x'], coordinates[node]['y']))
        for outfall in outfalls:
            X.add_node(outfall, pos=(coordinates[outfall]['x'], coordinates[outfall]['y']))
        for storage in storages:
            X.add_node(storage, pos=(coordinates[storage]['x'], coordinates[storage]['y']))
        for cunduit in conduits:
            inlet = conduits[cunduit]['FromNode']
            outlet = conduits[cunduit]['ToNode']
            if cunduit in ['JTN-03(tb2).1','JTN-03(tb).2']:
                pass
            else:
                X.add_edge(inlet, outlet,length=conduits[cunduit]['Length'],
                           diameter=inp.XSECTIONS[cunduit]['Geom1'] if cunduit in inp.XSECTIONS else 0,
                           slop=(conduits[cunduit]['InOffset']-conduits[cunduit]['OutOffset'])/conduits[cunduit]['Length'])

        node_ids = list(coordinates.keys())

        out_storage_nodes = list(outfalls.keys()) + list(storages.keys()) + ['J04030201110701158368','JJTNfoulout','JDG-11','JJTN-03(tb2)']##需要排除的节点
        first_labels = np.where(nse < 0)[0]
        second_labels = np.where(nse < 0.5)[0]
        best_labels = np.where(nse >= 0.5)[0]

        first_node = {f'{node_ids[i]}' for i in first_labels}

        second_node = [f'{node_ids[i]}' for i in second_labels if f'{node_ids[i]}' not in out_storage_nodes]
        second_node_index = [i for i in second_labels if f'{node_ids[i]}' not in out_storage_nodes]

        best_node = [f'{node_ids[i]}' for i in best_labels if f'{node_ids[i]}' not in out_storage_nodes]
        best_node_index = [i for i in best_labels if f'{node_ids[i]}' not in out_storage_nodes]

        chosen_indices = np.random.choice(len(best_node),len(best_node),replace=False)

        junct_nodes = second_node+[best_node[indices] for indices in chosen_indices]
        junct_index = second_node_index+[best_node_index[indices] for indices in chosen_indices]

        unmask_node = {f'{node_ids[i]}' for i in label_nodes}
        coordinates_list = np.array([[coordinates[coor]['x'], coordinates[coor]['y']] for coor in coordinates])
        from scipy.spatial import distance
        information_dict ={}
        for ini_ix,start_node in enumerate(junct_nodes):
            print(start_node)
            distances = distance.cdist([coordinates_list[node_ids.index(start_node)]],coordinates_list[label_nodes],'euclidean')
            distances = np.squeeze(distances,axis=0)
            sorted_list = np.sort(distances)
            for nearest in sorted_list:
                nearest_node_index = np.where(distances == nearest)[0]
                target_node = node_ids[label_nodes[nearest_node_index][0]]
                if nx.has_path(X, source=start_node, target=target_node) or nx.has_path(X, source=target_node, target=start_node):
                    edge_us = self.edges_up_with_depth(X, start_node, target_node)
                    edge_ds = self.edges_ds_with_depth(X, start_node, target_node)
                    if not edge_us :
                        target_depth = [('edge_ds', edge_ds[-1][-1])]
                    elif not edge_ds :
                        target_depth = [('edge_us', edge_us[-1][-1])]
                    else:
                        target_depth = [('edge_us', edge_us[-1][-1]) if edge_us[-1][0] == target_node else ('edge_ds', edge_ds[-1][-1])]

                    information_dict[start_node]={'edge_us':edge_us,'edge_ds':edge_ds,'target_depth':target_depth}
                    break
                else:
                    continue
            downstream_path = self.find_downstream_path_no_branches(X, start_node)
            upstream_path = self.find_upstream_path_no_branches(X, start_node)
            single_branch = downstream_path + [start_node] + upstream_path
            information_dict[start_node].update({'downstream_path':downstream_path,'upstream_path':upstream_path,
                                                 'single_branch':single_branch,'start_node':start_node,
                                                 'target_node':target_node,'start_node_index':junct_index[ini_ix]
                                                    ,'subcatchments':subcatchments[start_node]})
            # [index for index, value in enumerate(edge_ds) if value[1] == 'J04030201020700699643']

        return information_dict,unmask_node

    def edges_up_with_depth(self,G,start_node,target_node):
        visited = set()
        edge_us = []
        # edge_weights = []
        stack = [(start_node, 0)]  # (current_node, current_depth)
        while stack:
            node, depth = stack.pop()
            visited.add(node)
            for pred in G.predecessors(node):
                length = G.edges[pred, node]['length']
                diameter = G.edges[pred, node]['diameter']
                slop = G.edges[pred, node]['slop']
                edge_us.append((pred, node, length, diameter, slop, depth))
                if pred == target_node:
                    break
                else:
                    if pred not in visited:
                        stack.append((pred, depth + 1))
        return edge_us
    def edges_ds_with_depth(self,G,start_node,target_node):
        visited = set()
        edge_ds = []
        # edge_weights = []
        stack = [(start_node, 0)]  # (current_node, current_depth)
        while stack:
            node, depth = stack.pop()
            visited.add(node)
            for pred in G.successors(node):
                length = G.edges[node, pred]['length']
                diameter = G.edges[node, pred]['diameter']
                slop = G.edges[node, pred]['slop']
                edge_ds.append((node, pred, length, diameter, slop, depth))
                if pred == target_node:
                    break
                else:
                    if pred not in visited:
                        stack.append((pred, depth + 1))
        return edge_ds

    def find_upstream_path_no_branches(self,G, start_node):
        path = []
        current_node = start_node
        while True:
            predecessors = list(G.predecessors(current_node))
            if len(predecessors) != 1:
                pred_len = [len(list(G.predecessors(find_node))) for find_node in predecessors]
                if sum(pred_len)>1:
                    break

            # 获取上一个节点
            if len(predecessors)==0:
                break
            else:
                prev_node = predecessors[0]

            # 检查上一个节点是否有多个后继节点
            if len(list(G.successors(prev_node))) != 1:
                path.append(prev_node)
                break

            # 添加上一个节点到路径
            path.append(prev_node)
            current_node = prev_node

        return path
    def find_downstream_path_no_branches(self,G, start_node):
        path = []
        current_node = start_node

        while True:
            successors = list(G.successors(current_node))

            if len(successors) != 1:
                break
            next_node = successors[0]

            stop_outer_loop = False
            if len(list(G.predecessors(next_node))) != 1:
                find_nodes = [nodes for nodes in list(G.predecessors(next_node)) if nodes != current_node]
                for find_node in find_nodes:
                    if not list(G.predecessors(find_node)):
                        continue
                    else:
                        stop_outer_loop = True
                        path.append(next_node)
                        break
                if stop_outer_loop:
                    break

            path.append(next_node)
            current_node = next_node

        return path
