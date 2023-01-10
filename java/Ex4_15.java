public class Ex4_15 {
    public static void main(String[] args){
        int number = 12321;
        int tmp = number;
        int result = 0;

        String temp = "";

        while(tmp!=0){
            temp += (tmp%10); // temp는 숫자를 거꾸로 반환한 값을 저장
            tmp /= 10; // tmp = tmp/10;
        }

        result = Integer.parseInt(temp);

        if(number==result){
            System.out.println(number+"는 회문수입니다.");
        } else {System.out.println(number+"는 회문수가 아닙니다.");}

    }
}
