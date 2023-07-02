# **CHORD**

## **join_protocal**

### **1. join 요청**

  * join reply를 받고 successor의 ID와 addr을 저장
  * fingertable의 모든 값을 successor의 ID와 addr로 init
### **2. join 요청처리**

* **내가 DHT상에 혼자인 경우**
  * successor ID = req ID
  * fingertable의 0번째에 req ID 삽입
  * req에게 내 ID, address를 전달(reply)

* **내 ID &#60; req ID &#60;&#61; successor ID 인 경우**
  * successor ID = req ID
  * fingertable의 0번째에 req ID 삽입
  * req에게 이전 successor의 ID, address를 전달(reply)
  
* **else**
  * successor에게 req의 ID, address를 join하도록 요청
  
###### *&nbsp; &nbsp; # req : join을 요청한 node*
###### *&nbsp; &nbsp; ## linked list에 노드 추가하기와 비슷함*
&nbsp;
## **stabilize_protocal**

###### *&nbsp; &nbsp; # 항상 돌아가는 background method*

### **1. stabilize()**
* **내 ID &#60; from ID &#60;&#61; successor ID 인 경우**
  * successor ID = from ID
  * fingertable의 0번째에 from ID 삽입

내 ID, address를 successor에게 (이전노드라고) 전달( **NOTIFY**, predecessor를 갱신하도록함 )\
fingertable에 저장된 주소에 index (self.id+2**n)와 자신의 주소를 보내서\
**SUCCESSOR**(get_successor())를 수행하도록함

### **2. get_successor()**

* **내 ID &#60; req ID &#60;&#61; successor ID 인 경우**
  * req ID, successor ID & addr를 다시 보냄( **SUCCESSOR_REP** )
  
* **predecessor가 없거나,  predecessor ID &#60; req ID &#60;&#61; 내 ID 인 경우**
  * req ID, 내 ID & addr를 다시 보냄( **SUCCESSOR_REP** )
  
* **else**
  * **SUCCESSOR** 요청을 그대로 successor에게 보냄

###### *&nbsp; &nbsp; # SUCCESSOR_REP를 받는 req node는 받은 정보로 fingertable을 갱신함*
