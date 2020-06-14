"""
Package Hierarchy :
패키지 하이어라키

BaseComponenet : 프로젝트 베이스 클래스
↑ DataManager : 데이터 조작을 위한 도구들 모음 (pad_sequencing, tokenizing)
↑ DataGenerator : 분리된 데이터파일들을 검증하고 하나의 통합된 파일 생성
↑ DataBuilder : 데이터를 로드하고 tensor로 변형해서 학습 가능한 형태로 만듬
"""
