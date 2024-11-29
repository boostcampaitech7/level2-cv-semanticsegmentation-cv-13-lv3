
# 📖 Overview
![](https://i.imgur.com/SqupAoR.png)
Bone Segmentation은 인공지능 분야에서 중요한 응용 분야 중 하나로, 특히, 딥러닝 기술을 이용한 뼈 Segmentation은 많은 연구가 이루어지고 있으며, 다양한 목적으로 도움을 줄 수 있습니다.

<br>

## 🗂 Dataset
- **Input :** hand bone x-ray 객체가 담긴 이미지가 모델의 인풋으로 사용됩니다. segmentation annotation은 json file로 제공됩니다.
- **Output :** 모델은 각 픽셀 좌표에 따른 class를 출력하고, 이를 rle로 변환하여 리턴합니다. 이를 output 양식에 맞게 csv 파일을 만들어 제출합니다.
<br><br>
- 전체 이미지 개수: 800장(Train), 288장(Test)
- 크게 손가락 / 손등 / 팔로 구성되며, 총 29개의 class (뼈 종류)가 존재합니다.
<br><br><br>

## 📃 Metric
![image](https://github.com/user-attachments/assets/f77da0ea-caf8-4e15-a592-dab7f6c331b0)
2 * (예측 영역 ∩ 실제 영역) / (예측 영역의 크기 + 실제 영역의 크기)인 DICE score는 예측된 영역과 실제 영역 간의 중첩 정도를 수치화하여 표현합니다. 
이 공식은 두 영역이 완전히 일치할 때 최대값인 1을 갖고, 전혀 겹치지 않을 때 최소값인 0을 갖습니다.




<!-- - **Annotations :** Image size, class,  -->

<!-- <br/> -->

<br><br>
# Team CV-13

## 🧑‍💻 Members 
<table>
    <tr height="160px">
        <td align="center" width="150px">
            <a href="https://github.com/boyamie"><img height="80px"  src="https://github.com/user-attachments/assets/adeaf63c-a763-46df-bd49-1a0ce71098eb"></a>
            <br/>
            <a href="https://github.com/boyamie"><strong>김보현</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/Ja2Hw"><img height="80px"  src="https://github.com/user-attachments/assets/d824f102-e0a5-491d-9c75-cb90f625da3e"/></a>
            <br/>
            <a href="https://github.com/Ja2Hw"><strong>김재환</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/Jin-SukKim"><img height="80px"  src="https://github.com/user-attachments/assets/f15196cd-96fa-404c-b418-dc84e5ced92a"/></a>
            <br/>
            <a href="https://github.com/Jin-SukKim"><strong>김진석</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/202250274"><img height="80px" src="https://github.com/user-attachments/assets/534a7596-2c95-4b89-867d-839a7728303c"/></a>
            <br />
            <a href="https://github.com/202250274"><strong>박진영</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/Superl3"><img height="80px" src="https://github.com/user-attachments/assets/3673ecc7-399b-42b0-9d94-cfcfd32d3864"/></a>
            <br />
            <a href="https://github.com/Superl3"><strong>성기훈</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
              <a href="https://github.com/hocheol0303"><img height="80px"  src="https://github.com/user-attachments/assets/2d0a71c6-9752-43a8-b96e-bc3be06e5dde"/></a>
              <br />
              <a href="https://github.com/hocheol0303"><strong>양호철</strong></a>
              <br />
          </td>
    </tr>
</table>  
      
                

</br>

## 💻 Development Environment

- GPU : Tesla V100-SXM2-32GB
- 개발 언어: Python
- 프레임워크: Pytorch, Numpy, Pytorch Lightning, MMSegmentation
- 협업 툴: Github, Slack, Zoom


</br>

# Usage
## Pytorch Lightning
1. constants.py 파일 경로 수정
- 학습에 필요한 데이터셋 디렉토리 경로를 constants.py 파일에서 설정합니다
2. configs에서 설정 파일 작성
- SMP 라이브러리를 사용하려면 lightningmodule/configs 디렉토리 내에서 적절한 config 파일을 작성합니다.
3. 학습 실행 명령어
- 전체 데이터셋 학습
    ```
    python train.py
    ```
- 검증 포함 학습 (Group K-Fold 방식 사용)
    ```
    python train.py --validation
    ```
- 특정 config 파일 사용
    ```
    python train.py --config "path_to_config_file"
    ```
4. 테스트(Test)
- Checkpoint 파일로 추론
    ```
    python test.py
    ```
- PT 파일로 추론
    ```
    python test.py --pt
    ```
- 손바닥 Crop 데이터셋을 사용한 모델 추론
    ```
    python test.py --palm
    ```
- 앙상블 추론
    ```
    python test.py --ensemble
    ```

<br><br>


# 🔦 Models
- 프로젝트에서 사용된 주요 이미지 세그멘테이션 모델의 특징 및 구현 라이브러리

| Model     | Feature                                                                                     | Library                |
|------------|-----------------------------------------------------------------------------------------------|-----------------------------|
| **DeepLab v3** | Atrous convolution을 사용해 다양한 스케일에서 객체를 효과적으로 분리. ASPP 모듈로 해상도 다양성 통합. | Segmentation Models Pytorch |
| **U-Net++**   | 복잡한 네트워크 구조와 개선된 스킵 연결을 통해 세그멘테이션 정확도 향상.                         | mmsegmentation             |
| **HRNet**    | 다양한 해상도에서 이미지를 병렬로 처리하며, 해상도 간 정보 교환을 통한 세밀한 세그멘테이션 달성.    | mmsegmentation             |

<br><br>
