import torch
import os

class Setting(object):
    def __init__(self):
        DB = 'hematopoiesis' #MCA_subset #CITE_ASAP #ms #PBMC #brain 
        self.use_cuda = False
        self.threads = 1

        if not self.use_cuda:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:0')

       
        self.graph_contrastive_weight = 1
        self.graph_contrastive_tau = 0.3
        self.graph_pos_threshold = 0.8
        self.graph_pos_symmetric = True
        self.lambda_adv = 0.08
        self.adv_warmup_epochs = 5
        if DB == "BMMC_1":
            self.number_of_class = 22 # Number of cell types in scRNA-seq
            self.input_size = 11362
            self.rna_paths = ['Data/Data-1/rna_1_filter.h5ad'] # scRNA-seq data
            self.atac_paths = ['Data/Data-1/atac_1_filter.h5ad'] # scATAC-seq data
            self.rna_protein_paths = [] # Protein expression from reference data
            self.atac_protein_paths = [] # Protein expression from target data
            self.peak_paths = ['Data/Data-1/peak_1_PCA50.h5ad'] #peak_1_tfidf
            self.atac_labels = True
            
            # Training setting            
            self.batch_size = 256
            self.lr = 0.008
            self.lr_decay_epoch = 30  # 延迟学习率衰减，给更多时间学习
            self.epochs = 30  # 增加训练轮数（22类细胞，对齐难度大）
            self.embedding_size = 64
            self.momentum = 0.9
            self.seed = 1
            self.checkpoint = ''
        if DB == "BMMC_verfiy":
            self.number_of_class = 22 # Number of cell types in scRNA-seq
            self.input_size = 11362
            self.rna_paths = ['Data/Data-1_partial/rna_1_filter.h5ad'] # scRNA-seq data
            self.atac_paths = ['Data/Data-1_partial/atac_1_filter.h5ad'] # scATAC-seq data
            self.rna_protein_paths = [] # Protein expression from reference data
            self.atac_protein_paths = [] # Protein expression from target data
            self.peak_paths = ['Data/Data-1_partial/peak_1_PCA50.h5ad'] #peak_1_tfidf
            self.atac_labels = True
            
            # Training setting            
            self.batch_size = 256
            self.lr = 0.008
            self.lr_decay_epoch = 30  # 延迟学习率衰减，给更多时间学习
            self.epochs = 30  # 增加训练轮数（22类细胞，对齐难度大）
            self.embedding_size = 64
            self.momentum = 0.9
            self.seed = 1
            self.checkpoint = ''    
            
        if DB == "BMMC_2":
            self.number_of_class = 30 
            self.input_size = 11407
            self.rna_paths = ['Data/Data-2/rna_2_filter.h5ad']
            self.atac_paths = ['Data/Data-2/atac_2_filter.h5ad']
            self.rna_protein_paths = []
            self.atac_protein_paths = []
            self.peak_paths = ['Data/Data-2/peak_2_PCA50.h5ad']
            self.atac_labels = True
            
            # Training setting            
            self.batch_size = 256
            self.lr = 0.008
            self.lr_decay_epoch = 30
            self.epochs = 30
            self.embedding_size = 64
            self.momentum = 0.9
            self.seed = 1
            self.checkpoint = ''
            
            
        if DB == "BMMC_3":
            self.number_of_class = 21 
            self.input_size = 11428 
            self.rna_paths = ['Data/Data-3/rna_3_filter.h5ad'] 
            self.atac_paths = ['Data/Data-3/atac_3_filter.h5ad'] 
            self.rna_protein_paths = [] 
            self.atac_protein_paths = [] 
            self.peak_paths = ['Data/Data-3/peak_3_PCA50.h5ad']
            self.atac_labels = True
            
            # Training setting            
            self.batch_size = 256
            self.lr = 0.008 
            self.lr_decay_epoch = 30
            self.epochs = 30
            self.embedding_size = 64
            self.momentum = 0.9
            self.seed = 1
            self.checkpoint = ''
            
            
        if DB == "BMMC_4":
            self.number_of_class = 30
            self.input_size = 11450
            self.rna_paths = ['Data/Data-4/rna_4_filter.h5ad']
            self.atac_paths = ['Data/Data-4/atac_4_filter.h5ad']
            self.rna_protein_paths = []
            self.atac_protein_paths = []
            self.peak_paths = ['Data/Data-4/peak_4_PCA50.h5ad']
            self.atac_labels = True
            
            # Training setting            
            self.batch_size = 256
            self.lr = 0.008
            self.lr_decay_epoch = 30
            self.epochs = 30
            self.embedding_size = 64
            self.momentum = 0.9
            self.seed = 1
            self.checkpoint = ''
            
        
        if DB == "CITE_ASAP":
            self.number_of_class = 7 
            self.input_size = 17219 # Number of common genes and proteins between reference data and target data
            self.rna_paths = ['Data/Data-7/adata_ref_rna.h5ad']
            self.atac_paths = ['Data/Data-7/adata_tar_atac.h5ad']
            self.rna_protein_paths = ['Data/Data-7/adata_ref_adt.h5ad']
            self.atac_protein_paths = ['Data/Data-7/adata_tar_adt.h5ad']
            self.peak_paths = ['Data/Data-7/adata_tar_peak.h5ad'] #adt
            self.atac_labels = True
            
            # Training setting            
            self.batch_size = 256
            self.lr = 0.008
            self.lr_decay_epoch = 30
            self.epochs = 30
            self.embedding_size = 64
            self.momentum = 0.9
            self.seed = 1
            self.checkpoint = ''
            
        if DB == "MCA_subset":
            self.number_of_class = 14 
            self.input_size = 17057 
            self.rna_paths = ['Data/Data-6/adata_mca_gem.h5ad']
            self.atac_paths = ['Data/Data-6/adata_mca_gam.h5ad']
            self.rna_protein_paths = [] 
            self.atac_protein_paths = [] 
            self.peak_paths = ['Data/Data-6/adata_mca_peak.h5ad']
            self.atac_labels = True
            
            # Training setting            
            self.batch_size = 256 
            self.lr = 0.008
            self.lr_decay_epoch = 30
            self.epochs = 30
            self.embedding_size = 64
            self.momentum = 0.9
            self.seed = 1
            self.checkpoint = ''
            
        if DB == "ms":
            self.number_of_class = 10 
            self.input_size = 11055 
            self.rna_paths = ['Data/Data-5/ms_rna.h5ad']
            self.atac_paths = ['Data/Data-5/ms_atac.h5ad']
            self.rna_protein_paths = [] 
            self.atac_protein_paths = [] 
            self.peak_paths = ['Data/Data-5/ms_PCA50.h5ad']
            self.atac_labels = True
            
            # Training setting            
            self.batch_size = 256
            self.lr = 0.008 
            self.lr_decay_epoch = 30  # 更新为最优值（基于sweep结果）
            self.epochs = 30
            self.embedding_size = 64
            self.momentum = 0.9
            self.seed = 1
            self.checkpoint = ''
    
            
        # if DB == "brain":
        #     self.number_of_class = 23
        #     self.input_size = 17403
        #     self.rna_paths = ['data_brain/brain_rna.h5ad']
        #     self.atac_paths = ['data_brain/brain_atac.h5ad']
        #     self.rna_protein_paths = [] 
        #     self.atac_protein_paths = [] 
        #     self.peak_paths = ['data_brain/brain_PCA50.h5ad']
        #     self.atac_labels = False
            
        #     # Training setting            
        #     self.batch_size = 256
        #     self.lr = 0.008
        #     self.lr_decay_epoch = 30
        #     self.epochs = 30
        #     self.embedding_size = 64
        #     self.momentum = 0.9
        #     self.seed = 1
        #     self.checkpoint = '' 
            
        # if DB == "hf":
        #     self.number_of_class = 15
        #     self.input_size = 17187
        #     self.rna_paths = ['data_hf/hf_rna.h5ad'] 
        #     self.atac_paths = ['data_hf/hf_atac_data_1k.h5ad'] 
        #     self.rna_protein_paths = [] 
        #     self.atac_protein_paths = [] 
        #     self.peak_paths = ['data_hf/hf_PCA50.h5ad']
        #     self.atac_labels = False
            
        #     # Training setting            
        #     self.batch_size = 256
        #     self.lr = 0.008
        #     self.lr_decay_epoch = 30
        #     self.epochs = 30
        #     self.embedding_size = 64
        #     self.momentum = 0.9
        #     self.seed = 1
        #     self.checkpoint = ''
        if DB == "PBMC":
            self.number_of_class = 19
            self.input_size = 18353
            self.rna_paths = ['Data/pbmc_10x/adata_rna_common_genes.h5ad'] 
            self.atac_paths = ['Data/pbmc_10x/adata_atac_gam_common_genes.h5ad'] 
            self.rna_protein_paths = [] 
            self.atac_protein_paths = [] 
            self.peak_paths = ['Data/pbmc_10x/pbmc_atac_lsi.h5ad']
            self.atac_labels = True
            
            # Training setting            
            self.batch_size = 256
            self.lr = 0.008
            self.lr_decay_epoch = 30  # 稍微延迟学习率衰减
            self.epochs = 30  # 增加训练轮数以获得更好的收敛
            self.embedding_size = 64
            self.momentum = 0.9
            self.seed = 1
            self.checkpoint = ''
         
        if DB == "ms_sub":
            self.number_of_class = 29
            self.input_size = 15519
            self.rna_paths = ['Data/data_subset/adata_rna_facs.h5ad'] 
            self.atac_paths = ['Data/data_subset/adata_atac_cache.h5ad'] 
            self.rna_protein_paths = [] 
            self.atac_protein_paths = [] 
            self.peak_paths = ['Data/data_subset/MCAsubset_atac_tsne.h5ad']
            self.atac_labels = True
            
            # Training setting            
            self.batch_size = 256
            self.lr = 0.008
            self.lr_decay_epoch = 30
            self.epochs = 30
            self.embedding_size = 64
            self.momentum = 0.9
            self.seed = 1
            self.checkpoint = ''    
        if DB == "ms_os":
            self.number_of_class = 29
            self.input_size = 15519
            self.rna_paths = ['Data/ms_os/adata_rna_facs.h5ad'] 
            self.atac_paths = ['Data/ms_os/adata_atac_cache.h5ad'] 
            self.rna_protein_paths = [] 
            self.atac_protein_paths = [] 
            self.peak_paths = ['Data/ms_osMCAOS_atac_tsne.h5ad']
            self.atac_labels = True
            
            # Training setting            
            self.batch_size = 256
            self.lr = 0.008
            self.lr_decay_epoch = 30
            self.epochs = 30
            self.embedding_size = 64
            self.momentum = 0.9
            self.seed = 1
            self.checkpoint = ''   
        if DB == "hematopoiesis":
            self.number_of_class = 9
            self.input_size = 15714
            self.rna_paths = ['Data/hematopoiesis/hematopoiesis_rna.h5ad'] 
            self.atac_paths = ['Data/hematopoiesis/hematopoiesis_atac.h5ad'] 
            self.rna_protein_paths = [] 
            self.atac_protein_paths = [] 
            self.peak_paths = ['Data/hematopoiesis/hematopoiesis_atac_pca50.h5ad']
            self.atac_labels = True
            
            # Training setting            
            self.batch_size = 256
            self.lr = 0.008
            self.lr_decay_epoch = 30
            self.epochs = 30
            self.embedding_size = 64
            self.momentum = 0.9
            self.seed = 1
            self.checkpoint = ''
         
         # ============================================================
        if DB == "adbrainCortex":
            # Unique cell_type strings in obs (13; two clusters share name "L4")
            self.number_of_class = 13
            # Must match rna.h5ad / atac.h5ad n_vars (scJoint common genes = 3088)
            self.input_size = 3088
            self.rna_paths = ['Data/adbrainCortex/adbrainCortex_rna.h5ad'] 
            self.atac_paths = ['Data/adbrainCortex/adbrainCortex_atac.h5ad'] 
            self.rna_protein_paths = [] 
            self.atac_protein_paths = [] 
            self.peak_paths = ['Data/adbrainCortex/adbrainCortex_atac_pca50.h5ad']
            self.atac_labels = True
            
            # Training setting            
            self.batch_size = 256
            self.lr = 0.008
            self.lr_decay_epoch = 30
            self.epochs = 30
            self.embedding_size = 64
            self.momentum = 0.9
            self.seed = 1
            self.checkpoint = ''

        if DB == "endo":
            # Must match unique obs['cell_type'] in endo_rna.h5ad (currently 16)
            self.number_of_class = 16
            self.input_size = 24438
            self.rna_paths = ['Data/endo/endo_rna.h5ad']
            self.atac_paths = ['Data/endo/endo_atac.h5ad']
            self.rna_protein_paths = []
            self.atac_protein_paths = []
            self.peak_paths = ['Data/endo/endo_atac_pca50.h5ad']
            self.atac_labels = True

            self.batch_size = 256
            self.lr = 0.008
            self.lr_decay_epoch = 30
            self.epochs = 30
            self.embedding_size = 64
            self.momentum = 0.9
            self.seed = 1
            self.checkpoint = ''

        if DB == "pairkidney":
            self.number_of_class = 13
            self.input_size = 19459
            self.rna_paths = ['Data/kidney/kidney_rna.h5ad']
            self.atac_paths = ['Data/kidney/kidney_atac.h5ad']
            self.rna_protein_paths = []
            self.atac_protein_paths = []
            self.peak_paths = ['Data/kidney/kidney_atac_pca50.h5ad']
            self.atac_labels = True

            self.batch_size = 256
            self.lr = 0.008
            self.lr_decay_epoch = 30
            self.epochs = 30
            self.embedding_size = 64
            self.momentum = 0.9
            self.seed = 1
            self.checkpoint = ''

        if DB == "bcc":
            self.number_of_class = 19
            self.input_size = 17582
            self.rna_paths = ['Data/bcc/bcc_rna.h5ad']
            self.atac_paths = ['Data/bcc/bcc_atac.h5ad']
            self.rna_protein_paths = []
            self.atac_protein_paths = []
            self.peak_paths = ['Data/bcc/bcc_atac_pca50.h5ad']
            self.atac_labels = True

            self.batch_size = 256
            self.lr = 0.008
            self.lr_decay_epoch = 30
            self.epochs = 2
            self.embedding_size = 64
            self.momentum = 0.9
            self.seed = 1
            self.checkpoint = ''
        if DB == "unpairkidney":
            # RNA reference has 15 types; ATAC eval uses 11 shared types after build filtering.
            self.number_of_class = 15
            self.input_size = 28143
            self.rna_paths = ['Data/unpairkidney/unpairkidney_rna.h5ad']
            self.atac_paths = ['Data/unpairkidney/unpairkidney_atac.h5ad']
            self.rna_protein_paths = []
            self.atac_protein_paths = []
            self.peak_paths = ['Data/unpairkidney/unpairkidney_atac_pca50.h5ad']
            self.atac_labels = True


            self.batch_size = 256
            self.lr = 0.008
            self.lr_decay_epoch = 30
            self.epochs = 30
            self.embedding_size = 64
            self.momentum = 0.9
            self.seed = 1
            self.checkpoint = ''
        if DB == "heart":
            self.number_of_class = 12
            self.input_size = 19237
            self.rna_paths = ['Data/heart/heart_rna.h5ad']
            self.atac_paths = ['Data/heart/heart_atac.h5ad']
            self.rna_protein_paths = [] 
            self.atac_protein_paths = [] 
            self.peak_paths = ['Data/heart/heart_atac_pca50.h5ad']
            self.atac_labels = True
            
            # Training setting            
            self.batch_size = 256
            self.lr = 0.008
            self.lr_decay_epoch = 30
            self.epochs = 30
            self.embedding_size = 64
            self.momentum = 0.9
            self.seed = 1
            self.checkpoint = ''
