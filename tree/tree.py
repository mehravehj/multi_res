import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical as Categorical


# Python program to create a Complete Tree from
# its linked list representation
# since the tree is small in our case we generate the whole tree at once rather than adding leaves along the way

# Linked List node
# Python program to create a Complete Binary Tree from
# its linked list representation

# Linked List node
class List_Node:

	# Constructor to create a new node
	def __init__(self, data):
		self.data = data
		self.previous = None
		self.next = None


class node_Data:

	# Constructor to create a new node
	# initial probability is zero
	def __init__(self, myId, probability=0):
		self.probability = probability
		self.UID = myId  # nice to have


# Binary Tree Node structure
class Binary_Tree_Node:

	# Constructor to create a new node
	def __init__(self, data):
		self.data = data
		self.first_Child = None
		self.next_Sibling = None

		self.childId = None  # 1-4

		self.parent = None

	def print_Me(self):
		# just ro traverse over tree
		print("*******")
		print("Node " , self.data.UID)
		print("With probability of traversal " , self.data.probability)

		if self.parent != None:
			print("I am the number " + str(self.childId) + " of Node " + str(self.parent.data.UID))
		else:
			print("I am the ROOT!")

		if self.next_Sibling != None:
			print("My next sibling is " , self.next_Sibling.data.UID)
		else:
			print("I don't have a next sibling")

		if self.first_Child != None:
			print("My first child is " , self.first_Child.data.UID)
		else:
			print("I don't have children!")
		print("-------")


# Various options for defining our tree
class Tree_Config:
	# Constructor to create a new node
	def __init__(self, children_Per_Parent=4, generations=10): # basically children are resolutions and generations are layers
		self.generations = generations
		self.children_Per_Parent = children_Per_Parent

	def print_Me(self):
		print("Tree configuration")
		print("Children per parent = ", self.children_Per_Parent)
		print("Generations = ", self.generations)


# Class to convert the linked list to Binary Tree
class Quaternary_Tree:

	# Constructor for storing head of linked list
	# and root of the Binary Tree
	def __init__(self):
		self.list_Head = None
		self.list_Tail = None
		self.tree_Root = None # this is the input image
		self.config = None

	def init_Tree(self, tree_Config):
		self.config = tree_Config

		if self.config == None:
			print("Quaternary_Tree.initTree: *****ERROR***** NO CONFIG LOADED")
			return
		else:
			print("Tree configuration loaded:")
			self.config.print_Me()
		print(self.config.children_Per_Parent)
		print(self.config.generations)
		target_Amount = self.getNodesAmount(self.config.children_Per_Parent, self.config.generations)
		curr_ChildId = 1
		node_Id = 0
		curr_Node = None
		curr_ParentNode = None
		previous_Child = None

		node_Queue = []  # temporary node storage as we generate the tree

		# root generation
		curr_Node = self.generate_New_Base_TreeNode(node_Id, curr_ChildId)
		node_Id += 1

		self.tree_Root = curr_Node

		node_Queue.append(curr_Node)

		while node_Id < target_Amount:

			# get Next Parent
			curr_Parent = node_Queue.pop(0)

			while curr_ChildId < self.config.children_Per_Parent + 1:
				# create new node
				curr_Node = self.generate_New_Base_TreeNode(node_Id, curr_ChildId)

				# link node properly

				curr_Node.parent = curr_Parent  # parent link
				if previous_Child != None:
					previous_Child.next_Sibling = curr_Node  # sibling link
				if curr_ChildId == 1:
					curr_Parent.first_Child = curr_Node  # first child link

				# update loop variables
				node_Id += 1
				curr_ChildId += 1
				previous_Child = curr_Node
				node_Queue.append(curr_Node)

			# we have created children_Per_Parent children.
			# reset loop variables

			curr_ChildId = 1
			previous_Child = None

	def getNodesAmount(self, children_Per_Parent, generations):
		amount = 0
		curr_Generation = 0

		while curr_Generation < generations + 1:
			amount = amount + pow(children_Per_Parent, curr_Generation)
			curr_Generation += 1

		return amount

	def generate_New_Base_TreeNode(self, node_Id, childId):
		data = node_Data(node_Id)
		node = Binary_Tree_Node(data)
		node.childId = childId

		return node

	# Just update the linked list with new data
	def push(self, new_data):

		# Create the node
		newList_Node = List_Node(new_data)

		if self.list_Tail == None:
			self.list_Tail = newList_Node

		# Update the list with the new List_Node aka setting the next
		newList_Node.next = self.list_Head

		if self.list_Head != None:
			self.list_Head.previous = newList_Node

		# Update Conversion Object state
		self.list_Head = newList_Node

	# This might explode  if tree is too big
	def inorder_Traversal(self, node):
		if (node):
			self.inorder_Traversal(node.first_Child)
			node.print_Me()
			self.inorder_Traversal(node.next_Sibling)

	def breadth_First_Traversal(self, node):
		node_Queue = []

		node_Queue.append(node)
		curr_Node = None

		while node_Queue:
			curr_Node = node_Queue.pop(0)

			curr_Node.print_Me()

			curr_Node = curr_Node.first_Child

			while curr_Node != None:
				node_Queue.append(curr_Node)

				curr_Node = curr_Node.next_Sibling

	def get_Children(self, parentNode):
		child_List = []
		curr_Node = parentNode.first_Child

		while curr_Node != None:
			child_List.append(curr_Node)

			curr_Node = curr_Node.next_Sibling


		return child_List


def sample_with_probability(tree): # sample based on probability
	path = []
	path_id = []
	node = tree.tree_Root
	for gen in range(tree.config.generations):
		children = tree.get_Children(node)
		children_prob = torch.FloatTensor([c.data.probability for c in children])
		prob = nn.functional.softmax(children_prob, dim=0)
		prob = Categorical(prob)
		sampled_child = int(prob.sample().data)
		node = children[sampled_child]
		path.append(node)
		path_id.append(node.childId)
	sampling_function = [x1 - x2 for (x1, x2) in zip([c.childId for c in path], [c.parent.childId for c in path])]
	return path, path_id, sampling_function


def update_path_prob(path, new_value, mixing_weight=0.1): # Update probabilities with mixing weights: x = (1-mixw) * x_old + mixw * x_new
	for p in path:
		p.data.probability = (1-mixing_weight) * p.data.probability + mixing_weight *new_value


# Driver Program to test above function

# Object of conversion class
# conv = Conversion()
# conv = Quaternary_Tree()
# conv.initTree(Tree_Config(3, 3))
#
# # print("Inorder Traversal of the contructed Tree is:")
# # conv.inorder_Traversal(conv.tree_Root)
# # conv.breadth_First_Traversal(conv.tree_Root)
#
#
# root = conv.tree_Root
# print(root)
# root.probability = 1
# print(root.probability)
# children = conv.get_Children(root)
# prob = [c.data.probability for c in children]
# print(prob)
# print(conv.config.generations)

