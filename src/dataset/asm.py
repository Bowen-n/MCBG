# created by Wu Bolun
# 2020.9.29
# bowenwu@sjtu.edu.cn

import os
import re
import json
import pickle
from typing import List, Dict, Set
from collections import OrderedDict

import networkx as nx

from constant import *


class Instruction(object):
    ''' Assembly Instruction '''
    def __init__(self, address, 
                 opcode: str, operands: List[str]) -> None:
        super(Instruction, self).__init__()

        if isinstance(address, str):
            self.address = int(address, 16)
        else:
            self.address = address
        self.opcode = opcode
        self.operands = list(map(lambda x: x.strip(','), operands))
        self.next_addr = int()
        
        # tags
        self.start: bool = False
        self.branchto: int = None
        self.fallthrough: bool = True
        self.call: bool = False
        self.ret: bool = False
    
        # opcode type
        # only consider jmp/cjmp for control flow changing
        if self.opcode in JMP_OPCODES:
            self.optype = 'jmp'
        elif self.opcode in CJMP_OPCODES:
            self.optype = 'cjmp'
        elif self.opcode in END_OPCODES:
            self.optype = 'end'
        else:
            self.optype = 'regular'

        self.addr_pattern = re.compile(r'[0-9A-Fa-f]{4,}')
    
    def _parse_var_arg(self, vars_dict):
        ''' convert vars args to value '''
        def __map_func(operand):
            if operand.startswith('[') and operand.endswith(']'):
                parsed_operand = ''
                current_item = ''
                for char in operand:
                    if char == '[':
                        parsed_operand += char
                    elif char in ['+', '-', ']']:
                        if current_item in vars_dict:
                            parsed_operand += vars_dict[current_item]
                        else:
                            parsed_operand += current_item
                        parsed_operand += char
                        current_item = ''
                    else:
                        current_item += char
                return parsed_operand   
            else:
                return operand
        
        self.operands = list(map(__map_func, self.operands))

    def accept(self, parser) -> None:
        if self.optype == 'jmp':
            parser.visit_jump(self)
        elif self.optype == 'cjmp':
            parser.visit_conditional_jump(self)
        elif self.optype == 'end':
            parser.visit_end(self)
        else:
            parser.visit_default(self)

    def find_dst_addr(self):
        ''' Get destination address for jmp/cjmp
        e.g.
            jnz     short loc_542360
            jb      short $+2
        '''
        # first situation
        for item in self.operands:
            match = self.addr_pattern.search(item)
            if match is None:
                continue
            else:
                dst_addr = int(match.group(), 16)
                return dst_addr

        # second situation
        for item in self.operands:
            if '$+' in item:
                append = int(item.split('+')[-1])
                return self.address + append

        return None

    def dict_format(self) -> dict:
        instruction = {}
        instruction['address'] = self.address
        instruction['opcode'] = self.opcode
        instruction['operands'] = self.operands
        instruction['next_addr'] = self.next_addr
        return instruction

    def __repr__(self) -> str:
        ''' overload `print()` '''
        instruction = '{}: {}'.format(hex(self.address), self.opcode)
        for operand in self.operands:
            instruction += ' {}'.format(operand)
        instruction += ' next:{}'.format(hex(self.next_addr))
        return instruction  


class Block(object):
    ''' Basic block '''
    def __init__(self) -> None:
        super(Block, self).__init__()
        self.start_addr: int = -1
        self.end_addr: int = -1

        self.insn_list: List[Instruction] = []
        self.in_edge_list: List[int] = []
        self.out_edge_list: List[int] = []
    
    def dict_format(self) -> dict:
        block = {}
        block['start_addr'] = self.start_addr
        block['end_addr'] = self.end_addr
        
        insns = []
        for insn in self.insn_list:
            insns.append(insn.dict_format())

        block['insn_list'] = insns
        block['in_edge_list'] = self.in_edge_list
        block['out_edge_list'] = self.out_edge_list
        return block

    def __repr__(self) -> str:
        ''' overload `print()` '''
        block = '----------BLOCK AT {}---------\nINSTRUCTIONS\n'.format(hex(self.start_addr))
        for insn in self.insn_list:
            block += '\t{}\n'.format(insn)
        block += 'IN_EDGES\n'
        for edge in self.in_edge_list:
            block += '\t{}\n'.format(hex(edge))
        block += 'OUT_EDGES\n'
        for edge in self.out_edge_list:
            block += '\t{}\n'.format(hex(edge))
        block.rstrip('\n')
        return block


class AsmParser(object):
    ''' Parse .asm file for BIG2015 Dataset
    1. Convert to one-to-one mapping from sorted addresses to assembly instructions
    2. Extract CFG(Control Flow Graph)
    '''
    def __init__(self, directory: str, binary_id: str, ) -> None:
        super(AsmParser, self).__init__()

        # file
        self.binary_id = binary_id
        self.filepath = os.path.join(directory, binary_id+'.asm')

        # re pattern
        self.byte_pattern = re.compile(r'^[A-F0-9?][A-F0-9?]\+?$')

        # store
        self.assembly: OrderedDict[int, Instruction] = OrderedDict()
        self.blocks: OrderedDict[int, Block] = OrderedDict()

        # var, arg lookup
        self.vars_dict = {}

    ''' Main Functions '''
    def parse(self) -> bool:
        self.parse_instructions()
        if len(self.assembly.keys()) == 0:
            return False
        self.parse_blocks()
        self.clean_blocks()
        if len(self.blocks.keys()) == 0:
            return False
        return True

    def parse_instructions(self) -> None:
        ''' Preprocess the .asm file, convert to one-to-one mapping 
            from sorted addresses to assembly instructions,
            stored in `self.assembly`
        '''
        file_input = open(self.filepath, 'rb')

        prev_addr = -1

        count = 0
        for line in file_input:

            elems = line.split()
            decoded_elems = list(map(lambda x: x.decode('utf-8', 'ignore'), elems))
            if len(decoded_elems) == 0:
                continue
            
            # check if in code segment and get address
            seg = decoded_elems.pop(0)
            if len(decoded_elems) == 0:
                continue
            addr = self._get_address_from_seg(seg)
            if addr == 'NotInCodeSeg':
                continue
            
            # position assembly instruction
            insn_index = self._get_index_of_insn(decoded_elems)
            end_index = self._get_index_of_comment(decoded_elems)

            # No bytes found
            if insn_index == 0:
                if '=' in decoded_elems:
                    self.vars_dict[decoded_elems[0]] = decoded_elems[-1]
                continue
            if insn_index < end_index:
                insn_list = decoded_elems[insn_index: end_index]
                # deal with ??+_text
                if '+' in insn_list[0] or '_' in insn_list[0]:
                    insn_list.pop(0)
                
                opcode = insn_list[0]

                # pseudoinstruction
                if opcode in PSEUDO_OPCODES:
                    continue
                pseudo_ins = False
                for op in insn_list:
                    if op in PSEUDO_OPERANDS or '<' in op:
                        pseudo_ins = True
                        break
                if pseudo_ins == True:
                    continue

                # construct instruction
                addr = int(addr, 16)
                insn = Instruction(addr, opcode, insn_list[1:])
                insn._parse_var_arg(self.vars_dict)

                self.assembly[addr] = insn
                if prev_addr != -1:
                    self.assembly[prev_addr].next_addr = addr
                prev_addr = addr

                # print(insn)
        # print(self.assembly)
        # print(self.vars_dict)
        file_input.close()

    def parse_blocks(self) -> None:
        ''' Connect basic blocks to construct CFG
            Two iterations method which refers to
            http://www.cs.binghamton.edu/~ghyan/papers/dsn19.pdf
        '''
        # first iteration
        for addr, insn in self.assembly.items():
            insn.accept(self)

        # second iteration
        curr_block = None
        for addr, insn in self.assembly.items():
            if curr_block is None or insn.start == True:
                curr_block = self.get_block_at_addr(addr)
            
            if insn.next_addr in self.assembly:
                next_insn = self.assembly[insn.next_addr]
                if insn.fallthrough == True and next_insn.start == True:
                    next_block = self.get_block_at_addr(next_insn.address)
                    curr_block.out_edge_list.append(next_block.start_addr)
                    self.blocks[next_block.start_addr].in_edge_list.append(curr_block.start_addr)
            
            if insn.branchto is not None:
                block = self.get_block_at_addr(insn.branchto)
                # if it branches to the next instruction
                # edges have been dealt with previous logic
                if block.start_addr not in curr_block.out_edge_list:
                    curr_block.out_edge_list.append(block.start_addr)
                    self.blocks[block.start_addr].in_edge_list.append(curr_block.start_addr)
            
            curr_block.insn_list.append(insn)
            curr_block.end_addr = max(curr_block.end_addr, insn.address)
            self.blocks[curr_block.start_addr] = curr_block

    def clean_blocks(self) -> None:
        ''' Remove blocks which have neither in edges or out edges '''
        all_addrs = list(self.blocks.keys())
        for addr in all_addrs:
            if len(self.blocks[addr].in_edge_list) == 0 and \
               len(self.blocks[addr].out_edge_list) == 0:
               self.blocks.pop(addr)

    ''' Auxiliary Functions '''
    def get_block_at_addr(self, addr: int) -> Block:
        ''' Fetch block starts with `address`
        Ret:
            block_index - index in `self.block_list`
        '''
        if addr not in self.blocks:
            block = Block()
            block.start_addr  = addr
            block.end_addr = addr
            self.blocks[addr] = block
        
        return self.blocks[addr]

    def visit_jump(self, insn: Instruction) -> None:
        ''' Unconditional jump '''
        dst_addr = insn.find_dst_addr()
        self.assembly[insn.address].fallthrough = False
        if dst_addr is not None and dst_addr in self.assembly:
            self.assembly[insn.address].branchto = dst_addr
            self.assembly[dst_addr].start = True
        if insn.next_addr in self.assembly:
            self.assembly[insn.next_addr].start = True

    def visit_conditional_jump(self, insn: Instruction) -> None:
        ''' Conditional jump '''
        dst_addr = insn.find_dst_addr()
        if dst_addr is not None and dst_addr in self.assembly:
            self.assembly[insn.address].branchto = dst_addr
            self.assembly[dst_addr].start = True
        if insn.next_addr in self.assembly:
            self.assembly[insn.next_addr].start = True

    def visit_end(self, insn: Instruction) -> None:
        ''' End '''
        self.assembly[insn.address].fallthrough = False
        self.assembly[insn.address].ret = True
        if insn.next_addr in self.assembly:
            self.assembly[insn.next_addr].start = True

    def visit_default(self, insn: Instruction) -> None:
        pass

    def _get_index_of_insn(self, decoded_elems: List[str]) -> int:
        for i, elem in enumerate(decoded_elems):
            if not self.byte_pattern.match(elem):
                return i
        return i + 1

    def _get_index_of_comment(self, decoded_elems: List[str]) -> int:
        for i, elem in enumerate(decoded_elems):
            if elem.find(';') != -1:
                return i
            
        return len(decoded_elems)

    def _get_address_from_seg(self, seg: str) -> str:
        for codeseg in CODESEG_NAMES:
            if seg.startswith(codeseg) == True:
                colon = seg.rfind(':')
                if colon != -1:
                    return seg[colon+1:]
                else:
                    return seg[-8:]
        return "NotInCodeSeg"

    ''' Print & Store Functions '''
    def print_assembly(self) -> None:
        for addr, insn in self.assembly.items():
            print(insn)

    def print_blocks(self) -> None:
        count = 1
        for start_addr, block in sorted(self.blocks.items()):
            print('{}: {}'.format(count, block))
            count += 1
            
    def store_blocks(self, storepath, fformat='json') -> None:
        ''' fformat - 'json' / 'pickle'
        JSON
            blocks {int: block}
            block  {'start_adddr': int, 'end_addr': int, 'insn_list': list[insn]
                    'in_edge_list': list[int], 'out_edge_list': list[int]}
            insn   {'address': int, 'opcode': str, 'operands': list[str],
                    'next_addr': int}
        '''
        if fformat == 'pickle':
            with open(storepath, 'wb') as f:
                pickle.dump(self.blocks, f)
        elif fformat == 'json':
            
            blocks: Dict[int, Dict] = {}
            for addr, block in self.blocks.items():
                blocks[addr] = block.dict_format()

            with open(storepath, 'w') as f:
                json.dump(blocks, f)

# Scripts for test

if __name__ == '__main__':
    big2015_dir = '/home/wubolun/data/malware/big2015/train'
    binary_id = 'FhxiaMwrVAfXKq7NYkvU'
    # binary_id = 'LGbDxkN6wV9TedtYchBA'

    parser = AsmParser(directory=big2015_dir, binary_id=binary_id)
    parser.parse_instructions()
    # parser.print_blocks()
