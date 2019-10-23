UNASSIGNED = 'Unassigned'

def addNode(tree, string):
	tree[string] = UNASSIGNED 
	if '(' in string and ')' in string:
		nodes = findChildren(string)
		for node in nodes:
			tree = addNode(tree, node)
	return tree

def createTree(string):
	tree = {}
	tree = addNode(tree, string)
	return tree
	
def addParentInformation(tree):
	for node in tree:
		children = findChildren(node)
		for child in children:
			try:
				tree[child]['parent'] = node
			except:
				tree[child] = {}
				tree[child]['parent'] = node
	return tree
		
def renameNode(tree, nodeName, newName):
	outputTree = tree.copy()
	for otherNode in outputTree.keys():
		if ',' + nodeName + ':' in otherNode or '(' + nodeName + ':' in otherNode:
			del outputTree[otherNode]
			outputTree[otherNode.replace(nodeName, newName)] = UNASSIGNED 
		if findNodeName(otherNode) == nodeName:
			del outputTree[otherNode]
			outputTree[otherNode.replace(nodeName, newName)] = UNASSIGNED 
	return outputTree

def findBranchLength(node):
	branchLength = ''
	while not node == '':
		if not node[(len(node)-1)] == ';':
			branchLength = node[(len(node)-1)] + branchLength
		node = node[0:(len(node)-1)]
		if node[len(node)-1] == ':':
			return branchLength
	return None

def findNodeName(node):
	branchLength = ''
	fullNodeName = node
	while not node[len(node)-1] == ')':
		branchLength = node[(len(node)-1)] + branchLength
		node = node[0:(len(node)-1)]
		if not node == '':
			if node[len(node)-1] == ':':
				return node[0:(len(node)-1)]
		if node == '':
			return fullNodeName
	return fullNodeName

def findStructure(node):
	branchLength = ''
	while not node == '':
		branchLength = node[(len(node)-1)] + branchLength
		node = node[0:(len(node)-1)]
		if not node == '':
			if node[len(node)-1] == ')':
				return node			
	return ''

def findChildren(node):
	if '(' in node and ')' in node:
		children = []
		node = findStructure(node)
		node = node[0:len(node)-1]
		node = node[1:len(node)]
		bracketsOpen = 0
		bracketsClosed = 0
		counter = 0
		string = ''
		while counter < len(node):
			if node[counter] == ',' and bracketsOpen == bracketsClosed:
				children.append(string)
				string = ''
			elif node[counter] == '(':
				bracketsOpen = bracketsOpen + 1
				string = string + '('
			elif node[counter] == ')':
				bracketsClosed = bracketsClosed + 1
				string = string + ')'
			else:
				string = string + node[counter]
			counter = counter + 1		
		children.append(string)
		return children
	return []		

def findParent(tree, nodeName):
	for otherNode in tree:
		children = findChildren(otherNode)
		if not children == None:
			for child in children:
				if findNodeName(child) == nodeName or child == nodeName:
					return otherNode
	return None

def findDescendantNodes(node):
	tree = createTree(node)
	nodes = tree.keys()
	return [x for x in nodes if not x == node]

def dropDescendantNodes(tree, node, newName):
	outputTree = tree.copy()
	nodesToDrop = findDescendantNodes(node)
	for nodeToDrop in nodesToDrop:
		del outputTree[nodeToDrop]
	nodeName = findNodeName(node)
	outputTree = renameNode(outputTree, nodeName, newName)
	return outputTree

def dropNode(tree, node):
	parent = findParent(tree, node)
	if parent == None:
		del tree[node]
		return tree
	children = findChildren(parent)
	children = [x for x in children if not x == node]
	if len(children) > 0:
		newName = '(' + ','.join(children) + ')'
		newName = newName + findNodeNameWithoutStructure(parent)
		tree = dropDescendantNodes(tree, parent, newName)
		for child in children:
			tree = addNode(tree, child)
	if len(children) == 0:
		newName = findNodeNameWithoutStructure(parent)
		tree = dropDescendantNodes(tree, parent, newName)
	return tree

def findDescendantTips(node):
	descendants = findDescendantNodes(node)
	tips = []
	for descendant in descendants:
		if not '(' in descendant and not ')' in descendant:
			tips.append(descendant)
	return tips

def findTips(tree):
	tips = []
	for node in tree.keys():
		if not '(' in node and not ')' in node:
			tips.append(node)
	return tips

def findNodeNameWithoutStructure(node):	
	nodeName = findNodeName(node)
	structure = findStructure(node)
	result = nodeName.replace(structure, '')
	return result

def findRoot(tree):
	for node in tree:
		if ';' in node:
			return node
	return None

def findAncestors(tree, node):
	results = []
	for node2 in tree:
		if not node2 == node and node in node2:
			results.append(node2)
	return results

def findDepth(tree, node):
	ancestors = findAncestors(tree, node)
	depth = 0
	for ancestor in ancestors:
		branchLength = float(findBranchLength(ancestor))
		print branchLength
		depth = depth + branchLength
	return depth

def findMaximumHeight(node):
	children = findChildren(node)
	if children == []:
		return 0
	branchLength = float(findBranchLength(node))
	childrenMaximumHeights = []
	for child in children:
		maximumHeight = findMaximumHeight(child)
		branchLength = float(findBranchLength(child))
		childrenMaximumHeights.append(maximumHeight + branchLength)
	return max(childrenMaximumHeights)

def findAncestor(tree, node, numberAbove):
	current = node
	for i in xrange(numberAbove):
		current = tree[current]['parent']
	return current

def findAncestorWithSafeReturn(tree, node, numberAbove):
	current = node
	try:
		for i in xrange(numberAbove):
			current = tree[current]['parent']
	except:
		return current
	return current
	
def findRelatives(tree, tip, numberAbove):
	ancestor = findAncestorWithSafeReturn(tree, tip, numberAbove)
	descendantTips = findDescendantTips(ancestor)
	descendantTips.remove(tip)
	return descendantTips

def findGlottologId(node):
	nodeName = findNodeNameWithoutStructure(node)
	try:
		glottologId = nodeName.split('[')[1].split(']')[0]
		return glottologId 
	except:
		return None	
	
def saveTreeToFile(tree, fileName = None):
	string = str(tree)
	if not fileName == None:
		file = open(fileName, 'w')
		file.write(string)
	else:
		fileName = findNodeNameWithoutStructure(findRoot(tree)) + '.txt'
		file = open(fileName, 'w')
		file.write(string)

def readTreeFromFile(fileName):
	file = open(fileName, 'r').read()
	tree = eval(file)
	return tree
