import sys
import re
import glob
import json
import os.path

rx_call = re.compile(r'I bid (?P<call>[A-Z0-9]+)')
rx_card = re.compile(r'I play (?P<card>[A-Z][A-Z0-9])')

def board_iterator(fin):
	board = {}
	
	while True:
		line = next(fin).strip()
		if line.startswith('-x'):
			yield board
			return
		
		if line.startswith('-SWNE'):
			if len(board) > 0:
				yield board
				board = {}
		board['S'] = next(fin).strip()
		board['W'] = next(fin).strip()
		board['N'] = next(fin).strip()
		board['E'] = next(fin).strip()
		line = next(fin).strip()
		board['dealer'] = line[0].upper()
		board['vuln'] = 'ALL' if line[2:] == 'b' else line[2:].upper()

def gib_iterator(fin):
	auction = []
	explanations = []
	play = []
	result = None
	
	for line in fin:
		line = line.strip()
		if line.startswith('Result:'):
			result = line[8:]
			
			try:
				assert len(auction) == len(explanations)
				assert len(play) == 52 or set(auction) == {'P'}
			except:
				import pdb; pdb.set_trace()
			
			yield {
				'auction': auction,
				'explanations': explanations,
				'play': play,
				'result': result
			}
			
			auction, explanations, play, result = [], [], [], None
			continue
		
		m_call = rx_call.search(line)
		if m_call is not None:
			auction.append(m_call.groupdict()['call'])
			continue
		
		m_card = rx_card.search(line)
		if m_card is not None:
			play.append(m_card.groupdict()['card'])
			continue
			
		if line == 'That bid shows:':
			explanations.append('')
		
		if line.startswith('That bid shows: '):
			explanations.append(line[len('That bid shows: '):])
			
def combine_data(sets_glob_pattern, gib_folder):
	for fnm in sorted(glob.glob(sets_glob_pattern)):
		sys.stderr.write(fnm + '\n')
		
		gib_fnm = os.path.join(gib_folder, 'output.gib')
		board_it = board_iterator(open(fnm))
		gib_it = gib_iterator(open(gib_fnm))
		
		for board, gib in zip(board_it, gib_it):
			print('W:{}'.format(board['W']))
			print('N:{}'.format(board['N']))
			print('E:{}'.format(board['E']))
			print('S:{}'.format(board['S']))
			print('{} {}'.format(board['dealer'], board['vuln']))
			print(gib['result'])
			print('.'.join(gib['auction']))
			print(''.join(gib['play']))
			print(json.dumps(gib['explanations']))

if __name__ == '__main__':
	combine_data('input.gib', './')
