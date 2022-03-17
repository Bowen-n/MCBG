# created by Wu Bolun
# 2020.9.14
# bowenwu@sjtu.edu.cn

import os
import json
import time
import logging

import numpy as np
import networkx as nx


class R2GlobalCallGraphExtractor(object):
    ''' Extract opcode sequences of every function
        and the call relationship between functions
        Architecture: Radare2(https://radare.gitbooks.io/radare2book/content/)

        # Radare2 can parse incomplete binary (e.g. without PE header)
    '''
    def __init__(self, filepath):
        # package
        import r2pipe

        self.r2 = r2pipe.open(filepath, flags=['-2'])
        self.r2.cmd('aaa')
    

    def extract(self):
        # call graph json format
        call_graph = self.r2.cmdj('agCj')
        
        # get all functions
        functions = []
        for function in call_graph:
            if function['name'] not in functions:
                functions.append(function['name'])
            for func in function['imports']:
                if func not in functions:
                    functions.append(func)

        # get opcode sequences of all functions
        func_opseq = [] # [func_name, [opcode seq]]
        func_del = [] # function can't be disassembled
        for function in functions:
            # print(function)
            opcode_seq = []
            self.r2.cmd('s {}'.format(function))
            try:
                disassembly = self.r2.cmdj('pdfj')['ops']
            except:
                func_del.append(function)
                continue
            for instruction in disassembly:
                opcode_seq.append(instruction['type'])
            func_opseq.append([function, opcode_seq])
        
        # filt out functions can't be disassembled

        call_graph = list(filter(
            lambda x: x['name'] not in func_del, call_graph))
        for function in call_graph:
            function['imports'] = list(filter(
                lambda x: x not in func_del, function['imports']))
        
        opseq, adjmatrix = self._opseq_adjmatrix(func_opseq, call_graph)

        return opseq, adjmatrix

     
    def _opseq_adjmatrix(self, func_opseq, call_graph):
        func_count = len(func_opseq)
        func_order = {} # func index lookup dict
        for index in range(func_count):
            func_order[func_opseq[index][0]] = index

        # return
        ret_opseq = []
        ret_adjmatrix = np.zeros((func_count, func_count))

        for index in range(func_count):
            name = func_opseq[index][0]; opseq = func_opseq[index][1]
            ret_opseq.append(opseq)

            # get call relation of current function
            for content in call_graph:
                if content['name'] == name:
                    for call_func in content['imports']:
                        ret_adjmatrix[index][func_order[call_func]] = 1
                    break
        
        return np.array(ret_opseq, dtype=object), ret_adjmatrix


class AngrCFGExtractor(object):
    ''' Extract CFG(control flow graph) of a binary
        (1) CFG node - basic block
                edge - control flow
        (2) Architecture: Angr(https://docs.angr.io/)

        # Angr can only parse complete binary.
    
    '''
    def __init__(self, filepath):
        # package
        import angr

        self.filepath = filepath
        self.proj = angr.Project(filepath, load_options={'auto_load_libs': False})
        self.arch = self.proj.arch.name   # e.g. 'X86'
        self.mode = self.proj.arch.bits   # e.g. 32
        self.os = self.proj.loader.main_object.os

        # ignore logging
        logging.getLogger('angr').setLevel('CRITICAL')
        # logging.getLogger('angr').disabled = True


    def extract(self):
        ''' Extract CFG using CFGFast() '''
        
        start = time.time()
        self.cfg = self.proj.analyses.CFGFast(normalize=True)

        self.node_list = list(self.cfg.graph.nodes())
        self.edge_list = list(self.cfg.graph.edges())

        # output networkx Graph
        G = nx.DiGraph(arch=self.arch, mode=self.mode, os=self.os)
        for node in self.node_list:
            address = node.addr
            basic_block = self.proj.factory.block(address)
            assem_bytes = basic_block.bytes

            G.add_node(address, bytes=assem_bytes)
        
        for edge in self.edge_list:
            addr_src = edge[0].addr
            addr_tar = edge[1].addr

            G.add_edge(addr_src, addr_tar)
        
        end = time.time()
        
        print('Extract: {}, {}, {}'.format(self.filepath, self.arch, self.mode))
        print('Time: {}'.format(end-start))
        return G
        
