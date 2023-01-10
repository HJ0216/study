public class Ex4_14 {
    public static void main(String[] args){
        // 1~100 사이의 임의의 값을 얻어서 answer에 저장
        int answer = (int) (Math.random()*100)+1;
        int input = 0;
        int count = 0;

        java.util.Scanner s = new java.util.Scanner(System.in);

        do {
            count++;
            System.out.println("0~100 사이의 값을 입력하세요.");
            input = s.nextInt();

            if(answer==input){
                System.out.println("맞췄습니다.");
                System.out.println("시도 횟수: "+count);
                break;
            } else if(answer>input){
                System.out.println("더 큰 값을 입력하세요.");
            } else {System.out.println("더 작은 값을 입력하세요.");}
        } while(true);
    }
}

// do-while 반복문
// do{실행문} while(조건문): do 실행문 실행 후, while 조건문 확인
