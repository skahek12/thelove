# thelove
Integrated home cleaning management system

#프로젝트 개요
통합 홈 클리닝 관리 시스템
    - 실내 어지러움, 실내 공기 질, 쓰레기양을 종합한 지수인 “더럽 지수“로 모니터링 및 제어.
   1. 딥러닝을 이용한 실내 어지러움 측정
     - 홈 cctv의 실내 화면을 딥러닝(Deep learning)을 활용하여 실내 어지러움 정도를 모니터링 및 알람.
   2. 실내 공기 질 관리
     - 실내 먼지 양과 실외 대기상태 모니터링 후 창문 자동 개폐를 통한 실내 공기 질 관리.
   3. 스마트 쓰레기통
     - 쓰레기 양 모니터링 및 자동 압축.
   4. 더럽 지수
     - 실내 어지러움, 실내 공기 질, 쓰레기양 상태를 종합하여 한눈에 집 안 청결도를 확인 할 수 있다.

#파일설명
main.py & utill.py
이미지를 깔끔한지 어지러운지 딥러닝 CNN 기법으로 판별하고 수치화시켜 출력.

window_node_red.txt & trash_node_red.txt
node js 기반 node-red code
node-red 실행 후, txt 내용 import후 deploy
